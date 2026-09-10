"""W2NER decoding and entity-level metrics for contiguous BIO ocean entities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch


Entity = Tuple[int, int, str]


def decode_contiguous_w2ner(
    grid: torch.Tensor,
    length: int,
    id_to_label: Dict[int, str],
) -> Set[Entity]:
    """Recover contiguous spans from original W2NER NNW/THW labels.

    The copied ocean corpus only creates adjacent NNW links, so a valid entity
    requires all links from its head through its tail.  This is the exact label
    encoding used by ``ocean.py`` while avoiding the non-causal 2D CNN there.
    """
    result: Set[Entity] = set()
    labels = grid[:length, :length]
    for tail in range(length):
        for head in range(tail + 1):
            label_id = int(labels[tail, head])
            if label_id < 2:
                continue
            if tail > head and not all(int(labels[pos, pos + 1]) == 1 for pos in range(head, tail)):
                continue
            result.add((head, tail, id_to_label[label_id]))
    return result


def entity_prf(predictions: Sequence[Set[Entity]], gold: Sequence[Iterable[Tuple[int, int, int]]], id_to_label: Dict[int, str]):
    predicted_count = 0
    gold_count = 0
    correct_count = 0
    for predicted, gold_entities in zip(predictions, gold):
        expected = {(start, end, id_to_label[type_id]) for start, end, type_id in gold_entities}
        predicted_count += len(predicted)
        gold_count += len(expected)
        correct_count += len(predicted & expected)
    precision = correct_count / predicted_count if predicted_count else 0.0
    recall = correct_count / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
