"""Causal Qwen encoder and a causally scored W2NER relation head."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalW2NER(nn.Module):
    """Score W2NER cells without exposing a relation to unrestricted future text.

    For a cell ``(i, j)``, pair features use the structural state at
    ``max(i, j) + lookahead`` (clamped to the sentence end) and the representation
    at ``min(i, j)``.  Consequently all Qwen attention remains decoder-only.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int,
        state_size: int = 384,
        pair_size: int = 256,
        max_distance: int = 128,
        dropout: float = 0.15,
        lookahead: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.lookahead = lookahead
        self.max_distance = max_distance
        self.transition = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, state_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_gru = nn.GRU(state_size, state_size, batch_first=True)
        self.distance_embedding = nn.Embedding(max_distance + 1, 32)
        self.direction_embedding = nn.Embedding(3, 16)  # diagonal/lower/upper
        self.pair_head = nn.Sequential(
            nn.LayerNorm(state_size + hidden_size + 32 + 16),
            nn.Linear(state_size + hidden_size + 32 + 16, pair_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_size, num_labels),
        )
        self.end_head = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, 1),
        )

    @staticmethod
    def _gather_word_states(hidden: torch.Tensor, char_last_piece: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden.size(-1)
        indices = char_last_piece.unsqueeze(-1).expand(-1, -1, hidden_size)
        return hidden.gather(dim=1, index=indices)

    @staticmethod
    def _gather_positions(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        hidden_size = states.size(-1)
        return states.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_last_piece: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        word_states = self._gather_word_states(outputs.last_hidden_state, char_last_piece)
        word_states = word_states * word_mask.unsqueeze(-1)

        previous = F.pad(word_states[:, :-1], (0, 0, 1, 0))
        transition_input = torch.cat([word_states, word_states - previous], dim=-1)
        structural_states, _ = self.state_gru(self.transition(transition_input))
        structural_states = structural_states * word_mask.unsqueeze(-1)

        batch_size, length, _ = word_states.shape
        positions = torch.arange(length, device=word_states.device)
        row = positions.view(length, 1).expand(length, length)
        col = positions.view(1, length).expand(length, length)
        later = torch.maximum(row, col)
        earlier = torch.minimum(row, col)

        # A fixed confirmation delay remains causal: it only reads an already
        # consumed future prefix when a relation is emitted with that delay.
        valid_lengths = word_mask.long().sum(-1, keepdim=True).clamp_min(1)
        delayed = (later.unsqueeze(0) + self.lookahead).clamp_max(length - 1)
        delayed = torch.minimum(delayed, valid_lengths.unsqueeze(-1) - 1)
        later_states = self._gather_positions(
            structural_states,
            delayed.expand(batch_size, -1, -1).reshape(batch_size, -1),
        ).view(batch_size, length, length, -1)
        earlier_states = self._gather_positions(
            word_states,
            earlier.reshape(1, -1).expand(batch_size, -1),
        ).view(batch_size, length, length, -1)

        distance = (row - col).abs().clamp_max(self.max_distance)
        direction = torch.where(row == col, 0, torch.where(row > col, 1, 2))
        distance_features = self.distance_embedding(distance).unsqueeze(0).expand(batch_size, -1, -1, -1)
        direction_features = self.direction_embedding(direction).unsqueeze(0).expand(batch_size, -1, -1, -1)
        pair_features = torch.cat([later_states, earlier_states, distance_features, direction_features], dim=-1)
        pair_logits = self.pair_head(pair_features)

        # The completion head uses the same delayed state as a THW decision.
        end_positions = (positions.unsqueeze(0) + self.lookahead).clamp_max(length - 1)
        end_positions = torch.minimum(end_positions, valid_lengths - 1)
        end_states = self._gather_positions(structural_states, end_positions)
        end_logits = self.end_head(end_states).squeeze(-1)
        return {
            "pair_logits": pair_logits,
            "end_logits": end_logits,
            "word_states": word_states,
            "structural_states": structural_states,
        }


class CausalW2NERLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        none_weight: float = 0.15,
        boundary_weight: float = 0.5,
        commitment_weight: float = 0.2,
        margin: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.boundary_weight = boundary_weight
        self.commitment_weight = commitment_weight
        self.margin = margin
        weights = torch.ones(num_labels)
        weights[0] = none_weight
        self.register_buffer("class_weights", weights)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict) -> Dict[str, torch.Tensor]:
        pair_logits = outputs["pair_logits"]
        word_mask = batch["word_mask"]
        pair_mask = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)
        pair_loss = F.cross_entropy(
            pair_logits[pair_mask],
            batch["grid_labels"][pair_mask],
            weight=self.class_weights,
        )

        end_logits = outputs["end_logits"][word_mask]
        end_targets = batch["end_targets"][word_mask]
        positives = end_targets.sum().clamp_min(1.0)
        pos_weight = ((end_targets.numel() - positives) / positives).clamp(max=20.0)
        end_loss = F.binary_cross_entropy_with_logits(end_logits, end_targets, pos_weight=pos_weight)

        # For a gold span (s, e, type), the tail-head score at e must exceed
        # premature tail commitments at all earlier positions in the same span.
        commitment_terms = []
        for batch_index, entities in enumerate(batch["entities"]):
            for start, end, type_id in entities:
                if end <= start:
                    continue
                final_score = pair_logits[batch_index, end, start, type_id]
                prefix_scores = pair_logits[batch_index, start:end, start, type_id]
                commitment_terms.append(F.relu(self.margin + prefix_scores - final_score).mean())
        commitment_loss = torch.stack(commitment_terms).mean() if commitment_terms else pair_loss.new_zeros(())
        total = pair_loss + self.boundary_weight * end_loss + self.commitment_weight * commitment_loss
        return {
            "loss": total,
            "pair_loss": pair_loss.detach(),
            "end_loss": end_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
        }
