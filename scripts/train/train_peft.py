import os
import json
import time
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    IA3Config,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, required=True,
                   choices=["lora", "qlora", "dora", "ia3"])
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    return p.parse_args()


def get_target_modules():
    return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_target_modules(),
        )
        model = get_peft_model(model, peft_config)

    elif args.method == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.float16,
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_target_modules(),
        )
        model = get_peft_model(model, peft_config)

    elif args.method == "dora":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.float16,
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_target_modules(),
            use_dora=True,
        )
        model = get_peft_model(model, peft_config)

    elif args.method == "ia3":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.float16,
        )
        peft_config = IA3Config(
            task_type="CAUSAL_LM",
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
        model = get_peft_model(model, peft_config)

    else:
        raise ValueError(f"Unsupported method: {args.method}")

    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(args)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.train_file)["train"]

    def tokenize_fn(ex):
        tok = tokenizer(
            ex["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(args.method != "qlora"),
        bf16=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    start = time.time()
    peak_mem_before = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()

    end = time.time()
    total_minutes = (end - start) / 60.0

    peak_mem_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    trainable_params = 0
    total_params = 0
    for _, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    trainable_ratio = trainable_params / total_params if total_params > 0 else None

    summary = {
        "method": args.method,
        "model": args.model_name_or_path,
        "train_file": args.train_file,
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "epochs": args.num_train_epochs,
        "batch_size": args.per_device_train_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "seed": args.seed,
        "total_train_time_min": total_minutes,
        "peak_vram_gb": peak_mem_gb,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_ratio,
    }

    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Training finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
