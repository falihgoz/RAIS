import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from datasets.AudioDataset import AudioDataset
from configs.configs import Config
import numpy as np
from sklearn.model_selection import train_test_split
from avalanche.benchmarks import dataset_benchmark


def get_protocol_dir(dataset_name):
    base_path = Config.base_path

    protocol_paths = {
        "ASVspoof2019": (
            os.path.join(base_path, "ASVspoof2019/train_meta.txt"),
            os.path.join(base_path, "ASVspoof2019/dev_meta.txt"),
            os.path.join(base_path, "ASVspoof2019/eval_meta.txt"),
            None,
            torch.FloatTensor([0.1, 0.9]),
        ),
        "InTheWild": (
            os.path.join(base_path, "InTheWild/train_meta.txt"),
            os.path.join(base_path, "InTheWild/dev_meta.txt"),
            os.path.join(base_path, "InTheWild/eval_meta.txt"),
            None,
            torch.FloatTensor([0.5, 0.5]),
        ),
        "CFAD": (
            os.path.join(base_path, "CFAD/train_meta.txt"),
            os.path.join(base_path, "CFAD/dev_meta.txt"),
            os.path.join(base_path, "CFAD/eval_seen_unseen_meta.txt"),
            os.path.join(base_path, "CFAD/eval_unseen_meta.txt"),
            torch.FloatTensor([0.33, 0.67]),
        ),
        "VCC2020": (
            os.path.join(base_path, "VCC2020/train_meta.txt"),
            os.path.join(base_path, "VCC2020/dev_meta.txt"),
            os.path.join(base_path, "VCC2020/eval_meta.txt"),
            None,
            torch.FloatTensor([0.22, 0.78]),
        ),
        # OpenAI-LJSpeech
        "GPT": (
            os.path.join(base_path, "GPT/train_meta.txt"),
            os.path.join(base_path, "GPT/dev_meta.txt"),
            os.path.join(base_path, "GPT/eval_meta.txt"),
            None,
            torch.FloatTensor([0.5, 0.5]),
        ),
    }

    return protocol_paths.get(dataset_name)


def get_benchmark(args):
    dataset_list = "ASVspoof2019 VCC2020 InTheWild CFAD GPT"

    dataset_query = dataset_list.split()
    all_experience_train_datasets = []
    all_experience_dev_datasets = []
    all_experience_test_datasets = []
    all_dataset_weight = []

    for strate in dataset_query:
        print(f"load {strate} dataset...")
        (
            train_protocol_path,
            dev_protocol_path,
            eval_protocol_path,
            eval_unseen_protocol_path,
            dataset_weight,
        ) = get_protocol_dir(strate)

        train_dataset = AudioDataset(
            args,
            train_protocol_path,
            "train",
        )
        dev_dataset = AudioDataset(
            args,
            dev_protocol_path,
            "dev",
        )

        test_dataset = AudioDataset(
            args,
            eval_protocol_path,
            "eval",
        )

        print(f"Number of training data for {strate} dataset: {len(train_dataset)}")
        print(f"Number of dev data for {strate} dataset: {len(dev_dataset)}")
        print(f"Number of eval data for {strate} dataset: {len(test_dataset)}")
        all_experience_train_datasets.append(train_dataset)
        all_experience_dev_datasets.append(dev_dataset)
        all_experience_test_datasets.append(test_dataset)
        all_dataset_weight.append(dataset_weight)

    benchmark = dataset_benchmark(
        train_datasets=all_experience_train_datasets,
        test_datasets=all_experience_test_datasets,
        other_streams_datasets={"valid": all_experience_dev_datasets},
    )

    return benchmark, all_dataset_weight
