"""LoRA fine-tuning entry point for causal Qwen W2NER on the ocean corpus."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import OceanW2NERDataset, collate_w2ner, collect_types, read_bio
from src.metrics import decode_contiguous_w2ner, entity_prf
from src.model import CausalW2NER, CausalW2NERLoss


# Project root is resolved from this file, so PyCharm's working-directory
# setting does not affect the default data/output paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "weibo"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "qwen-weibo"))
    parser.add_argument("--max-chars", type=int, default=192)
    parser.add_argument("--max-pieces", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def autocast_context(device: torch.device, use_bf16: bool):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, loader, id_to_label, device, use_bf16):
    model.eval()
    predictions = []
    gold = []
    for batch in tqdm(loader, desc="Validate", leave=False):
        batch = move_batch(batch, device)
        with autocast_context(device, use_bf16):
            outputs = model(
                batch["input_ids"], batch["attention_mask"], batch["char_last_piece"], batch["word_mask"]
            )
        labels = outputs["pair_logits"].argmax(dim=-1).cpu()
        lengths = batch["word_mask"].sum(dim=-1).cpu().tolist()
        predictions.extend(
            decode_contiguous_w2ner(grid, length, id_to_label) for grid, length in zip(labels, lengths)
        )
        gold.extend(batch["entities"])
    return entity_prf(predictions, gold, id_to_label)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("--bf16 requires a CUDA device with bfloat16 support.")

    data_dir = Path(args.data_dir)
    train_examples = read_bio(data_dir / "example.train", args.max_chars)
    dev_examples = read_bio(data_dir / "example.dev", args.max_chars)
    test_path = data_dir / "example.test"
    if test_path.exists():
        test_examples = read_bio(test_path, args.max_chars)
        test_source = str(test_path)
    else:
        test_examples = dev_examples
        test_source = "example.dev (example.test not found)"
    types = collect_types(train_examples, dev_examples, test_examples)
    label_to_id = {label: index + 2 for index, label in enumerate(types)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    num_labels = len(types) + 2
    print(
        f"device={device}; train={len(train_examples)}; dev={len(dev_examples)}; "
        f"test={len(test_examples)} [{test_source}]; labels={label_to_id}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = OceanW2NERDataset(train_examples, tokenizer, label_to_id, args.max_pieces)
    dev_dataset = OceanW2NERDataset(dev_examples, tokenizer, label_to_id, args.max_pieces)
    collate = lambda items: collate_w2ner(items, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_dataset = OceanW2NERDataset(test_examples, tokenizer, label_to_id, args.max_pieces)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
    backbone = AutoModel.from_pretrained(args.model_name, torch_dtype=model_dtype, trust_remote_code=False)
    backbone.config.use_cache = False
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        backbone.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()
    model = CausalW2NER(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        num_labels=num_labels,
        lookahead=args.lookahead,
    ).to(device)
    criterion = CausalW2NERLoss(num_labels).to(device)

    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    update_steps = math.ceil(len(train_loader) / args.grad_accumulation)
    total_steps = update_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.bf16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"label_to_id": label_to_id, "num_labels": num_labels}
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_f1 = -1.0
    best_trainable_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        progress = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in progress:
            batch = move_batch(batch, device)
            with autocast_context(device, args.bf16):
                outputs = model(
                    batch["input_ids"], batch["attention_mask"], batch["char_last_piece"], batch["word_mask"]
                )
                loss_info = criterion(outputs, batch)
                loss = loss_info["loss"] / args.grad_accumulation
            scaler.scale(loss).backward()
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss_info["loss"])
            progress.set_postfix(loss=f"{loss_sum / step:.4f}", pair=f"{float(loss_info['pair_loss']):.3f}")

        metrics = evaluate(model, dev_loader, id_to_label, device, args.bf16)
        print(f"epoch={epoch} validation: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            # Keep only LoRA and task-head parameters in memory.  The frozen
            # Qwen base is not duplicated in the best-checkpoint snapshot.
            best_trainable_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            model.backbone.save_pretrained(output_dir / "adapter")
            tokenizer.save_pretrained(output_dir / "adapter")
            torch.save(
                {
                    "head": {key: value.cpu() for key, value in model.state_dict().items() if not key.startswith("backbone.")},
                    "best_f1": best_f1,
                },
                output_dir / "w2ner_head.pt",
            )
            print(f"saved best checkpoint to {output_dir} (F1={best_f1:.4f})")

    if best_trainable_state is not None:
        current_state = model.state_dict()
        for name, value in best_trainable_state.items():
            if name in current_state:
                current_state[name].copy_(value.to(current_state[name].device))
        model.load_state_dict(current_state, strict=False)

    test_metrics = evaluate(model, test_loader, id_to_label, device, args.bf16)
    print(
        "=" * 60
        + f"\nBest validation F1: {best_f1:.4f}"
        + f"\nTest source: {test_source}"
        + f"\nTest P={test_metrics['precision']:.4f} "
          f"R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f}"
        + "\n"
        + "=" * 60
    )


if __name__ == "__main__":
    main()
