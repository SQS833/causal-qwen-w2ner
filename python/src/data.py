"""Ocean BIO corpus loading and the W2NER grid encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


NNW_ID = 1


@dataclass
class Example:
    text: List[str]
    entities: List[Tuple[int, int, str]]  # inclusive character start/end/type


def read_bio(path: str | Path, max_chars: int) -> List[Example]:
    """Read the blank-line-separated ``character BIO-tag`` corpus from ocean.py."""
    examples: List[Example] = []
    blocks = Path(path).read_text(encoding="utf-8").strip().split("\n\n")
    for block in blocks:
        chars: List[str] = []
        entities: List[Tuple[int, int, str]] = []
        active_start: int | None = None
        active_type: str | None = None
        for raw_line in block.splitlines():
            fields = raw_line.split()
            if len(fields) < 2:
                continue
            char, tag = fields[0], fields[-1]
            if len(chars) >= max_chars:
                break
            index = len(chars)
            chars.append(char)
            if tag.startswith("B-"):
                if active_start is not None:
                    entities.append((active_start, index - 1, active_type or "UNK"))
                active_start, active_type = index, tag[2:]
            elif tag.startswith("I-") and active_start is not None:
                # The corpus uses contiguous BIO spans.  A type change closes the
                # old span and starts a recoverable new one.
                if tag[2:] != active_type:
                    entities.append((active_start, index - 1, active_type or "UNK"))
                    active_start, active_type = index, tag[2:]
            else:
                if active_start is not None:
                    entities.append((active_start, index - 1, active_type or "UNK"))
                    active_start, active_type = None, None
        if active_start is not None:
            entities.append((active_start, len(chars) - 1, active_type or "UNK"))
        if chars:
            examples.append(Example(chars, entities))
    return examples


def collect_types(*splits: Sequence[Example]) -> List[str]:
    return sorted({entity_type for split in splits for ex in split for _, _, entity_type in ex.entities})


class OceanW2NERDataset(Dataset):
    """Character-level dataset with Qwen pieces and original W2NER grid labels."""

    def __init__(self, examples: Sequence[Example], tokenizer, label_to_id: Dict[str, int], max_pieces: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_pieces = max_pieces

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        ex = self.examples[index]
        input_ids: List[int] = []
        char_last_piece: List[int] = []
        kept_chars: List[str] = []

        # Encoding character by character ensures every grid node has a causal
        # endpoint even if a tokenizer otherwise merges adjacent characters.
        for char in ex.text:
            piece_ids = self.tokenizer.encode(char, add_special_tokens=False)
            if not piece_ids:
                piece_ids = [self.tokenizer.unk_token_id]
            if len(input_ids) + len(piece_ids) > self.max_pieces:
                break
            input_ids.extend(piece_ids)
            char_last_piece.append(len(input_ids) - 1)
            kept_chars.append(char)

        length = len(kept_chars)
        grid = torch.zeros((length, length), dtype=torch.long)
        end_targets = torch.zeros(length, dtype=torch.float)
        valid_entities: List[Tuple[int, int, int]] = []
        for start, end, entity_type in ex.entities:
            if end >= length or entity_type not in self.label_to_id:
                continue
            type_id = self.label_to_id[entity_type]
            for position in range(start, end):
                grid[position, position + 1] = NNW_ID
            grid[end, start] = type_id
            end_targets[end] = 1.0
            valid_entities.append((start, end, type_id))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "char_last_piece": torch.tensor(char_last_piece, dtype=torch.long),
            "grid_labels": grid,
            "end_targets": end_targets,
            "entities": valid_entities,
            "text": "".join(kept_chars),
        }


def collate_w2ner(batch: Sequence[Dict], pad_token_id: int) -> Dict:
    batch_size = len(batch)
    max_pieces = max(item["input_ids"].numel() for item in batch)
    max_words = max(item["char_last_piece"].numel() for item in batch)
    input_ids = torch.full((batch_size, max_pieces), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_pieces), dtype=torch.long)
    char_last_piece = torch.zeros((batch_size, max_words), dtype=torch.long)
    word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
    grid_labels = torch.zeros((batch_size, max_words, max_words), dtype=torch.long)
    end_targets = torch.zeros((batch_size, max_words), dtype=torch.float)

    for row, item in enumerate(batch):
        piece_length = item["input_ids"].numel()
        word_length = item["char_last_piece"].numel()
        input_ids[row, :piece_length] = item["input_ids"]
        attention_mask[row, :piece_length] = 1
        char_last_piece[row, :word_length] = item["char_last_piece"]
        word_mask[row, :word_length] = True
        grid_labels[row, :word_length, :word_length] = item["grid_labels"]
        end_targets[row, :word_length] = item["end_targets"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "char_last_piece": char_last_piece,
        "word_mask": word_mask,
        "grid_labels": grid_labels,
        "end_targets": end_targets,
        "entities": [item["entities"] for item in batch],
        "texts": [item["text"] for item in batch],
    }
