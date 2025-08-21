#!/usr/bin/env python3
from typing import Callable, List, Optional, Union

import torch
from torch.optim import Optimizer
from packaging.version import parse
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training.utils import cycle
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

from numpy import inf
from typing import (
    Optional,
    List,
)

from torch.amp import GradScaler, autocast
from ais import AuxLabelBasedBuffer


class RAIS(SupervisedTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        mem_size: int = 200,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        fake_ratio=0.9,
        **base_kwargs,
    ):

        self.label_gen_output_masked = None
        self.label_gen_output_ori = None

        storage_policy = AuxLabelBasedBuffer(
            fake_ratio=fake_ratio,
            max_size=mem_size,
            model=model,
        )

        rais = RaisPlugin(
            mem_size=mem_size, batch_size=train_mb_size, storage_policy=storage_policy
        )
        if plugins is None:
            plugins = [rais]
        else:
            plugins += [rais]

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs,
        )

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        scaler = GradScaler()
        for self.i_batch, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            with autocast("cuda"):
                (
                    self.mb_output,
                    self.label_gen_output_masked,
                    self.label_gen_output_ori,
                ) = self.forward(self.mb_x, self.mbatch[2])
                self._after_forward(**kwargs)
                self.loss += self.losses()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)
            self.scaler.update()

            self._after_training_iteration(**kwargs)

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def compute_losses(
        self, classifier_logits, labels, label_gen_masked_logits, label_gen_ori_logits
    ):
        classifier_loss = self.criterion()
        distill_loss = F.mse_loss(label_gen_masked_logits, label_gen_ori_logits)

        avg_probs = label_gen_masked_logits.mean(dim=0)
        avg_probs = avg_probs.clamp(min=1e-9)
        uniform_probs = torch.full_like(
            avg_probs, 1.0 / label_gen_masked_logits.shape[1]
        )
        diversity_loss = F.kl_div(avg_probs.log(), uniform_probs, reduction="batchmean")

        selection_loss = (distill_loss + diversity_loss) / 2
        return classifier_loss, selection_loss

    def losses(self):
        if self.label_gen_output_masked is not None:
            classifier_loss, selection_loss = self.compute_losses(
                self.mb_output,
                self.mb_y,
                self.label_gen_output_masked,
                self.label_gen_output_ori,
            )
            total_loss = classifier_loss + selection_loss
        else:
            total_loss = self.criterion()

        return total_loss

    def check_label_frequencies(self, label_gen_output_masked, label_gen_output_ori):
        # Get the predicted labels (argmax over class probabilities)
        predicted_labels_masked = torch.argmax(label_gen_output_masked, dim=1)
        predicted_labels_ori = torch.argmax(label_gen_output_ori, dim=1)

        # Count frequency of each label in the batch
        unique_labels_masked, counts_masked = torch.unique(
            predicted_labels_masked, return_counts=True
        )
        unique_labels_ori, counts_ori = torch.unique(
            predicted_labels_ori, return_counts=True
        )

        # Sort by frequency (high to low) and print
        sorted_masked = sorted(
            zip(unique_labels_masked.tolist(), counts_masked.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_ori = sorted(
            zip(unique_labels_ori.tolist(), counts_ori.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        print("Frequencies in label_gen_output_masked (sorted):")
        for label, count in sorted_masked:
            print(f"Label {label}: {count}")

        print("\nFrequencies in label_gen_output_ori (sorted):")
        for label, count in sorted_ori:
            print(f"Label {label}: {count}")

    @torch.no_grad()
    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""

        self.model.eval()
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output, _, _ = self.forward(self.mb_x)

            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()
            self._after_eval_iteration(**kwargs)


class RaisPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Experience replay plugin.
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = strategy.train_mb_size
        batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None
        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args,
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)


__all__ = ["RAIS"]
