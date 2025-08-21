#!/usr/bin/env python3
from typing import List, Optional

import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.utils import cycle
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExemplarsSelectionStrategy,
    BalancedExemplarsBuffer,
)
from torch import Tensor
from torch.nn import Module

from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import DataLoader
from numpy import inf
from typing import (
    Dict,
    Optional,
    List,
    Set,
)
from avalanche.evaluation.metrics.eer_matrix import compute_eer

import numpy as np
from torch.amp import GradScaler, autocast
import pandas as pd


class AuxLabelBasedBuffer(BalancedExemplarsBuffer):
    """Stores samples for replay using a custom selection strategy and
    grouping."""

    def __init__(
        self,
        max_size: int,
        model: Module,
        fake_ratio=0.9,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param groupby: Grouping mechanism. One of {None, 'class', 'task',
            'experience'}.
        :param selection_strategy: The strategy used to select exemplars to
            keep in memory when cutting it off.
        """
        super().__init__(max_size)

        ss = AuxiliaryInformedSampling(
            model,
            fake_ratio,
        )

        self.selection_strategy = ss
        self.seen_groups: Set[int] = set()
        self._curr_strategy = None

    def post_adapt(self, agent, exp):
        new_data: AvalancheDataset = exp.dataset
        new_groups = self._split_by_experience(agent, new_data)
        self.seen_groups.update(new_groups.keys())

        # associate lengths to groups
        lens = self.get_group_lengths(len(self.seen_groups))
        group_to_len = {}
        for group_id, ll in zip(self.seen_groups, lens):
            group_to_len[group_id] = ll

        for group_id, new_data_g in new_groups.items():
            ll = group_to_len[group_id]
            new_buffer = ExperienceMemorySegment(ll, self.selection_strategy)
            new_buffer.update_from_dataset(agent, new_data_g)
            self.buffer_groups[group_id] = new_buffer

        # resize buffers
        for group_id, _ in self.buffer_groups.items():
            print("group_id: ", group_id)
            print("L: ", group_to_len[group_id])
            self.buffer_groups[group_id].resize(agent, group_to_len[group_id])

    def _split_by_experience(
        self, strategy, data: AvalancheDataset
    ) -> Dict[int, AvalancheDataset]:
        exp_id = strategy.clock.train_exp_counter + 1
        return {exp_id: data}


class ExperienceMemorySegment(ExemplarsBuffer):
    """A buffer that stores samples for replay using a custom selection
    strategy.

    This is a private class. Use `ParametricBalancedBuffer` with
    `groupby=None` to get the same behavior.
    """

    def __init__(
        self,
        max_size: int,
        selection_strategy: Optional["ExemplarsSelectionStrategy"] = None,
    ):
        """
        :param max_size: The max capacity of the replay memory.
        :param selection_strategy: The strategy used to select exemplars to
                                   keep in memory when cutting it off.
        """
        super().__init__(max_size)
        ss = selection_strategy
        self.selection_strategy = ss

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        new_data = strategy.experience.dataset
        self.update_from_dataset(strategy, new_data)

    def update_from_dataset(self, strategy, new_data):
        if len(self.buffer) == 0:
            self.buffer = new_data
        else:
            self.buffer = self.buffer.concat(new_data)

        idxs = self.selection_strategy.make_sorted_indices(
            strategy=strategy, data=self.buffer, max_size=self.max_size
        )

        self.buffer = self.buffer.subset(idxs[: self.max_size])

    def resize(
        self,
        strategy,
        new_size: int,
    ):
        self.max_size = new_size
        print("buffer len: ", len(self.buffer))
        if self.max_size <= len(self.buffer):
            self.buffer = self.buffer.subset(list(range(0, self.max_size)))
        else:
            self.buffer = self.buffer.subset(list(range(0, len(self.buffer))))


class AuxiliaryInformedSampling(ExemplarsSelectionStrategy):
    def __init__(
        self,
        model: Module,
        fake_ratio=0.8,
    ):
        super().__init__()
        self.label_generator = model
        self.fake_ratio = fake_ratio

    @torch.no_grad()
    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset, max_size=None
    ) -> List[int]:
        """Generates sorted indices based on features."""
        self.label_generator.eval()
        collate_fn = getattr(data, "collate_fn", None)
        self.label_generator.eval()
        binary_feat_list = []
        aux_feat_list = []

        for mb in DataLoader(
            data,
            collate_fn=collate_fn,
            batch_size=strategy.eval_mb_size,
            pin_memory=True,
            num_workers=0,
        ):
            # Call self.label_generator once
            output = self.label_generator(
                mb[0].to(strategy.device), mb[2].to(strategy.device)
            )

            binary_feat_list.append(output[0])  # Binary features (2D)
            aux_feat_list.append(output[1])  # Auxiliary features

        binary_feat = torch.cat(binary_feat_list, dim=0)  # Shape: (N, 2)
        aux_feat = torch.cat(aux_feat_list, dim=0)  # Shape: (N, K)

        return self.make_sorted_indices_from_features(
            binary_feat, aux_feat, max_size, self.fake_ratio, strategy
        )

    def stratify_and_sample(self, data, num_samples):
        grouped = data.groupby("label")

        group_iters = {
            label: group.sort_values(by="score", ascending=False).iterrows()
            for label, group in grouped
        }

        sampled_data = []
        remaining_samples = num_samples

        while remaining_samples > 0 and group_iters:
            for label in list(
                group_iters.keys()
            ):  # round-robin through remaining groups
                try:
                    _, sample = next(group_iters[label])
                    sampled_data.append(sample)
                    remaining_samples -= 1

                    if remaining_samples <= 0:  # Stop if we have enough samples
                        break
                except StopIteration:
                    group_iters.pop(label)  # Remove exhausted group

        return pd.DataFrame(sampled_data).reset_index(drop=True)

    def make_sorted_indices_from_features(
        self,
        binary_feat: Tensor,
        aux_feat: Tensor,
        max_size=None,
        ratio=0.9,
        strategy=None,
    ) -> List[int]:

        binary_feat = binary_feat.cpu().numpy()
        aux_feat = aux_feat.cpu().numpy()
        aux_labels = np.argmax(aux_feat, axis=1)

        aux_max_probs = np.max(aux_feat, axis=1)
        binary_max_probs = np.max(binary_feat, axis=1)

        confidence_score = (aux_max_probs + binary_max_probs) / 2

        unique_labels, counts = np.unique(aux_labels, return_counts=True)
        num_k = aux_feat.shape[1]

        assert num_k % 2 == 0, "n_label_size must be an even number."
        fake_label_range = (0, num_k // 2 - 1)  # First half is fake
        real_label_range = (num_k // 2, num_k - 1)  # Second half is real

        fake_indices = np.where(
            (aux_labels >= fake_label_range[0]) & (aux_labels <= fake_label_range[1])
        )[0]
        real_indices = np.where(
            (aux_labels >= real_label_range[0]) & (aux_labels <= real_label_range[1])
        )[0]

        fake_data = pd.DataFrame(
            {
                "index": fake_indices,
                "label": aux_labels[fake_indices],
                "score": confidence_score[fake_indices],
            }
        )
        real_data = pd.DataFrame(
            {
                "index": real_indices,
                "label": aux_labels[real_indices],
                "score": confidence_score[real_indices],
            }
        )

        num_fake_samples = int(max_size * ratio)
        num_real_samples = max_size - num_fake_samples

        # Track selected indices to avoid overlaps
        sampled_fake = self.stratify_and_sample(fake_data, num_fake_samples)
        sampled_real = self.stratify_and_sample(real_data, num_real_samples)

        # Combine fake and bona fide category and sort based on confidence score
        combined_samples = pd.concat([sampled_fake, sampled_real])
        combined_samples = combined_samples.sort_values(by="score", ascending=False)

        return combined_samples["index"].to_list()
