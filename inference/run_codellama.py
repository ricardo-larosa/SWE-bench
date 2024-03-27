import json
import logging
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk, load_dataset
from peft import PeftConfig, PeftModel
from tqdm.auto import tqdm
from transformers import (
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from llamao.modeling_flash_llama import LlamaForCausalLM as AutoModelForCausalLM
from make_datasets.utils import extract_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE_MAPS = json.load(open("codellama_device_maps.json"))


from pathlib import Path
from unsloth import FastLanguageModel

def get_output_file(
    output_dir,
    model_name_or_path,
    dataset_path,
    split,
    temperature,
    top_p,
    min_len,
    max_len,
    shard_id,
    num_shards,
):
    """
    Constructs the output file path based on the provided parameters.

    Args:
        output_dir (str): The directory where the output file will be saved.
        model_name_or_path (str): The name or path of the model.
        dataset_path (str): The path to the dataset.
        split (str): The dataset split.
        temperature (float): The temperature value.
        top_p (float): The top-p value.
        min_len (int): The minimum length of the output.
        max_len (int): The maximum length of the output.
        shard_id (int): The shard ID.
        num_shards (int): The total number of shards.

    Returns:
        str: The constructed output file path.
    """
    suffix = ""
    if min_len is not None:
        suffix += f"__min-{min_len}"
    if max_len is not None:
        suffix += f"__max-{max_len}"
    if shard_id is not None and num_shards is not None:
        suffix += f"__shard-{shard_id}-{num_shards}"
    if Path(dataset_path).exists():
        dset_nickname = Path(dataset_path).name + "__" + split
    else:
        dset_nickname = dataset_path.replace("/", "__") + "__" + split
    if Path(model_name_or_path).exists():
        if "checkpoint" in Path(model_name_or_path).name:
            model_nickname = Path(model_name_or_path).parent.name + "__" + Path(model_name_or_path).name
        else:
            model_nickname = Path(model_name_or_path).name
    else:
        model_nickname = model_name_or_path.replace("/", "__")
    output_file = Path(
        output_dir,
        dset_nickname
        + "__"
        + model_nickname
        + "__temp-"
        + str(temperature)
        + "__top-p-"
        + str(top_p)
        + suffix
        + ".jsonl",
    )
    if not output_file.parent.exists():
        output_file.parent.mkdir(
            parents=True, exist_ok=True
        )  # exists_ok=True for parallel
    return output_file


def load_model(model_name_or_path):
    """
    Loads a base model and optionally PEFT adapters.

    Args:
        model_name_or_path (str): The name or path of the base model.

    Returns:
        model: The loaded model.

    Raises:
        ValueError: If there is no device map for the specified model_name_or_path.
    """
    logger.info(f"Loading base model from {model_name_or_path}")
    max_memory = {
        **{
            k: f"{torch.cuda.get_device_properties(k).total_memory//1_010_000_000:d}GIB"
            for k in range(torch.cuda.device_count())
        },
        "cpu": "20GIB",
    }
    logger.info(f"Using max memory {max_memory}")
    if "-7b" in model_name_or_path:
        device_map = DEVICE_MAPS["7b"][str(torch.cuda.device_count())]
    elif "-13b" in model_name_or_path:
        device_map = DEVICE_MAPS["13b"][str(torch.cuda.device_count())]
    elif "-34b" in model_name_or_path:
        device_map = DEVICE_MAPS["34b"][str(torch.cuda.device_count())]
    else:
        raise ValueError(f"No device map for {model_name_or_path}")
    logger.info(f"Using device_map {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        max_memory=max_memory,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    ).eval()

    return model


def load_tokenizer(model_name_or_path):
    logger.info(f"Loading tokenizer {model_name_or_path}")
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def load_data(
    dataset_path, split, tokenizer, min_len, max_len, model_name_or_path, existing_ids, shard_id, num_shards
):
    """
    Load and preprocess the dataset for model inference.

    Args:
        dataset_path (str): The path to the dataset.
        split (str): The split of the dataset to load.
        tokenizer: The tokenizer used to tokenize the text.
        min_len (int): The minimum length of input sequences to include in the dataset.
        max_len (int): The maximum length of input sequences to include in the dataset.
        model_name_or_path (str): The name or path of the model.
        existing_ids: The list of existing instance IDs to filter out from the dataset.
        shard_id (int): The ID of the shard to load.
        num_shards (int): The total number of shards.

    Returns:
        dataset: The preprocessed dataset for model inference.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    if not Path(dataset_path).exists():
        dataset = load_dataset(dataset_path, split=split)
    elif Path(dataset_path, split).exists():
        dataset = load_from_disk(Path(dataset_path) / split)
    else:
        dataset = load_dataset(dataset_path)[split]
    if "input_ids" not in dataset.column_names:
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=False),
            batched=False,
            desc="tokenizing",
        )
    if "SWE-Llama" in model_name_or_path and dataset[0]["input_ids"][-2:] != [13, 13]:
        # SWE-Llama needs two exactly two newlines at the end
        dataset = dataset.map(
            lambda x: {"input_ids": x["input_ids"] + [13]}, batched=False
        )
    filter_func = None
    if min_len is not None and max_len is None:
        filter_func = lambda x: x >= min_len
    elif min_len is None and max_len is not None:
        filter_func = lambda x: x < max_len
    elif min_len is not None and max_len is not None:
        filter_func = lambda x: min_len <= x < max_len
    if filter_func is not None:
        dataset = dataset.filter(
            lambda x: filter_func(len(x["input_ids"])), desc="filtering for length"
        )
    lens = torch.tensor(list(map(lambda x: len(x["input_ids"]), dataset)))
    dataset = dataset.select(lens.argsort())
    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)
    dataset = dataset.filter(
        lambda x: x["instance_id"] not in existing_ids,
        desc="filtering for existing ids",
    )
    lens = torch.tensor(list(map(lambda x: len(x["input_ids"]), dataset)))  # recompute
    if shard_id is not None and num_shards is not None:
        logger.info(
            f"filtered dataset - {len(dataset)} examples, min length: {min(lens):_}, max length: {max(lens):_} (shard {shard_id} of {num_shards})"
        )
    else:
        logger.info(
            f"filtered dataset - {len(dataset)} examples, min length: {min(lens):_}, max length: {max(lens):_}"
        )
    return dataset


def generate(model, dataset, tokenizer, temperature, top_p, fileobj,model_name_or_path):
    class RepeatingTokensCriteria(StoppingCriteria):
        """
        Stopping criteria based on repeating tokens in the generated sequence.

        Attributes:
            min_length (int): The minimum length of the generated sequence.
            min_tokens (int): The minimum number of unique tokens required in the suffix of the generated sequence.
        """

        def __init__(self, min_length=100, min_tokens=10):
            super().__init__()
            self.min_length = min_length
            self.min_tokens = min_tokens

        def __call__(self, input_ids, scores, **kwargs):
            """
            Check if the stopping criteria is met based on repeating tokens.

            Args:
                input_ids (torch.Tensor): The input token IDs of the generated sequence.
                scores (torch.Tensor): The scores of the generated sequence.
                **kwargs: Additional keyword arguments.

            Returns:
                bool: True if the stopping criteria is met, False otherwise.
            """
            if input_ids[0, -1].cpu().item() == tokenizer.eos_token_id:
                return True
            if input_ids.shape[-1] < self.min_length:
                return False
            suffix = input_ids[0, -self.min_length:].cpu().tolist()
            if len(set(suffix)) <= self.min_tokens:
                return True
            return False

    stopping_criteria = StoppingCriteriaList([RepeatingTokensCriteria()])
    fail_count = 0
    with torch.no_grad():
        for ix, instance in enumerate(tqdm(dataset, desc=f"Generating patches")):
            try:
                input_ids = instance["input_ids"]
                input_ids = torch.tensor(
                    [input_ids], dtype=torch.long, device=model.device
                )
                logger.info(f"Processing {input_ids.shape[-1]} tokens")
                start = datetime.now()
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    temperature= 1.0 if temperature == 0 else temperature,
                    top_p=top_p,
                    do_sample=False if temperature == 0 else True,
                    max_new_tokens=200,
                    stopping_criteria=stopping_criteria,
                )
                total_len = output.shape[-1]
                output = output[0].cpu()[input_ids.shape[-1] :]
                new_len = len(output)
                logger.info(
                    f"Generated {new_len} tokens ({total_len} total) in {(datetime.now() - start).total_seconds()} seconds (speed: {new_len / (datetime.now() - start).total_seconds()} tps)"
                )
                output = tokenizer.decode(output, skip_special_tokens=False)
                logger.info(output[:200])
                diff = extract_diff(output)
                res = {
                    "instance_id": instance["instance_id"],
                    "full_output": output,
                    "model_patch": diff,
                    "model_name_or_path": Path(model_name_or_path).name,
                }
                print(json.dumps(res), file=fileobj, flush=True)
            except Exception as e:
                logger.exception(e)
                print(f"failed on {ix} with {len(input_ids)} tokens")
                fail_count += 1
                if fail_count >= 3:
                    raise ValueError("too many failures")


def get_all_existing_ids(output_file):
    stub_pattern = re.compile(r"((?:[\w\-\.]+)\_\_temp\-((\d+(\.\d+)?)|None)\_\_top\-p\-((\d+(\.\d+)?)|None))(\_\_|\.jsonl)")
    match = stub_pattern.match(output_file.name)
    if not output_file.exists():
        return set()
    if match is None:
        raise ValueError(f"output_file {output_file} doesn't match pattern")
    stub = match[1]
    existing_ids = set()
    output_files = list(Path(output_file.parent).glob(stub + "*"))
    for filename in output_files:
        logger.info(f"Loading existing ids from existing {filename}")
        with open(filename) as f:
            for line in f:
                datum = json.loads(line)
                existing_ids.add(datum["instance_id"])
    logger.info(f"Found {len(existing_ids)} existing ids")
    return existing_ids


def main(
    model_name_or_path,
    dataset_path,
    split,
    temperature,
    top_p,
    output_dir,
    min_len,
    max_len,
    shard_id,
    num_shards,
):
    if shard_id is not None and num_shards is None:
        raise ValueError("num_shards must be specified with shard_id")
    if shard_id is None and num_shards is not None:
        raise ValueError("shard_id must be specified with num_shards")
    output_file = get_output_file(
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path,
        split=split,
        temperature=temperature,
        top_p=top_p,
        min_len=min_len,
        max_len=max_len,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    logger.warning(f"output_file: {output_file}")
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/codellama-34b-bnb-4bit", # "codellama/CodeLlama-34b-hf" for 16bit loading
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        use_gradient_checkpointing = True,
        random_state = 3407,
        max_seq_length = max_seq_length,
    )
    existing_ids = get_all_existing_ids(output_file)
    dataset = load_data(
        dataset_path=dataset_path,
        split=split,
        tokenizer=tokenizer,
        min_len=min_len,
        max_len=max_len,
        model_name_or_path=model_name_or_path,
        existing_ids=existing_ids,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    with open(output_file, "a") as f:
        generate(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=top_p,
            fileobj=f,
            model_name_or_path=model_name_or_path,
        )

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to model or hf model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset or hf dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_len", type=int, default=None, help="Minimum length of input sequences to include")
    parser.add_argument("--max_len", type=int, default=None, help="Maximum length of input sequences to include")
    parser.add_argument("--shard_id", type=int, default=None, help="ID of the shard to load")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards")
    args = parser.parse_args()
    main(**vars(args))
