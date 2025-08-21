import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
import numpy as np
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    loss_metrics,
    EER_metrics,
)

from add import ADD
from avalanche.training import (
    Naive,
)

from rais import RAIS
from avalanche.training.plugins import (
    LRSchedulerPlugin,
    SupervisedPlugin,
)

from datasets.data_loader import get_benchmark
import argparse
from utils.utils import synchronize_inputs, EarlyStoppingPlugin
import random
from configs.configs import Config
import wandb
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True


def get_args():
    parser = argparse.ArgumentParser(
        description="Continual Learning for Audio Deepfake Detection"
    )
    # General parameters
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--max_audio_len", type=int, help="max audio length")
    parser.add_argument(
        "--buffer_size", type=int, default=512, help="Buffer size for replay"
    )
    parser.add_argument("--start_lr", type=float, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="weight decay"
    )
    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument("--early_stop", type=int, default=3, help="Early stop until")
    parser.add_argument(
        "--method",
        type=str,
        default="rais",
        help="Methods to use (space-separated)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for computation"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="aasist",
        choices=["aasist"],
        help="Audio deepfake detection networks",
    )
    parser.add_argument("--aux_lb_size", type=int, help="aux label size (k)")
    parser.add_argument("--dropout", type=float, help="dropout")
    parser.add_argument("--fake_ratio", default=0.8, type=float, help="r parameter")
    parser.add_argument(
        "--eval_all_exp",
        default="no",
        type=str,
        choices=["yes", "no"],
        help="eval all test exp for each exp",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.random_seed)

    # Continual learning strategy
    method_query = args.method.split()
    for strate in method_query:
        args = synchronize_inputs(args, strate)

        deepfake_benchmark, dataset_weights = get_benchmark(args)
        train_stream = deepfake_benchmark.train_stream
        valid_stream = deepfake_benchmark.valid_stream
        test_stream = deepfake_benchmark.test_stream

        print("========================================================")
        print("========================================================")
        print("current strate is {}".format(strate))
        print("Device: ", args.device)
        print("Number of experiences: ", len(train_stream))
        print("========================================================")
        print("========================================================")

        # logger
        interactive_logger = InteractiveLogger()
        text_logger = TextLogger()
        loggers = [interactive_logger]

        # Evaluator
        eval_plugin = EvaluationPlugin(
            EER_metrics(epoch=True, experience=True, stream=False),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=loggers,
        )

        early_stopping = EarlyStoppingPlugin(
            patience=args.early_stop,
            metric_name="Loss_Stream",
            val_stream_name="valid_stream",
            mode="min",
            peval_mode="epoch",
        )

        model = ADD(
            backbone=args.backbone,
            dropout=args.dropout,
            aux_label_size=args.aux_lb_size,
        )

        optimizer = Adam(
            model.parameters(),
            lr=args.start_lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            amsgrad=False,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0.000001
        )

        criterion = CrossEntropyLoss()

        if "rais" in strate:
            cl_strategy = RAIS(
                model,
                optimizer,
                criterion,
                train_epochs=args.epochs,
                train_mb_size=args.batch_size,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin,
                mem_size=args.buffer_size,
                device=args.device,
                plugins=[LRSchedulerPlugin(lr_scheduler), early_stopping],
                eval_every=1,
                fake_ratio=args.fake_ratio,
            )
        else:
            cl_strategy = Naive(
                model,
                optimizer,
                criterion,
                train_mb_size=args.batch_size,
                train_epochs=args.epochs,
                eval_mb_size=args.batch_size,
                evaluator=eval_plugin,
                device=args.device,
                peval_mode="epoch",
                plugins=[LRSchedulerPlugin(lr_scheduler), early_stopping],
                eval_every=1,
            )

        results = []

        for i, train_exp in enumerate(train_stream):
            print("Start of experience: ", train_exp.current_experience)
            print("Current Classes: ", train_exp.classes_in_this_experience)

            cl_strategy._criterion = CrossEntropyLoss(
                weight=dataset_weights[i].to(args.device)
            )
            res = cl_strategy.train(train_exp, eval_streams=[valid_stream[i]])

            if args.eval_all_exp == "no":
                results.append(cl_strategy.eval(test_stream[: (i + 1)]))
            else:
                results.append(cl_strategy.eval(test_stream))
    try:
        import wandb
        wandb.finish()

    except Exception as e:
        print("No active wandb run to close:", e)
