"""Run a saved LoRA Causal-W2NER checkpoint on one Chinese sentence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from src.data import Example, OceanW2NERDataset, collate_w2ner
from src.metrics import decode_contiguous_w2ner
from src.model import CausalW2NER


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "outputs" / "qwen-ocean"),
        help="Directory produced by train.py",
    )
    parser.add_argument("--text", default="中国科学院在北京开展海洋生态环境研究。")
    parser.add_argument("--end-threshold", type=float, default=0.0, help="Optional completion threshold; 0 disables it")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    label_to_id = {key: int(value) for key, value in config["label_to_id"].items()}
    id_to_label = {value: key for key, value in label_to_id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "adapter")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(config["model_name"], torch_dtype=dtype)
    backbone = PeftModel.from_pretrained(backbone, checkpoint / "adapter")
    model = CausalW2NER(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        num_labels=config["num_labels"],
        lookahead=config.get("lookahead", 0),
    ).to(device)
    head_state = torch.load(checkpoint / "w2ner_head.pt", map_location="cpu", weights_only=True)["head"]
    model.load_state_dict(head_state, strict=False)
    model.eval()

    dataset = OceanW2NERDataset([Example(list(args.text), [])], tokenizer, label_to_id, config["max_pieces"])
    batch = collate_w2ner([dataset[0]], tokenizer.pad_token_id)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        outputs = model(batch["input_ids"], batch["attention_mask"], batch["char_last_piece"], batch["word_mask"])
    grid = outputs["pair_logits"].argmax(dim=-1)[0].cpu()
    length = int(batch["word_mask"].sum())
    if args.end_threshold > 0:
        end_probability = torch.sigmoid(outputs["end_logits"][0]).cpu()
        for tail in range(length):
            if end_probability[tail] < args.end_threshold:
                for head in range(tail + 1):
                    if grid[tail, head] >= 2:
                        grid[tail, head] = 0
    entities = sorted(decode_contiguous_w2ner(grid, length, id_to_label))
    print(json.dumps(
        [{"text": args.text[start:end + 1], "start": start, "end": end, "type": entity_type}
         for start, end, entity_type in entities],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
