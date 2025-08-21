import yaml
import operator
import warnings
from copy import deepcopy
from avalanche.training.plugins import SupervisedPlugin


def load_config(config_file):
    with open(f"./configs/{config_file}", "r") as file:
        config = yaml.safe_load(file)
    return config


def synchronize_inputs(args, method):
    config_file = f"{method}.yaml"
    config = load_config(config_file)
    args.batch_size = (
        args.batch_size if args.batch_size is not None else config.get("batch_size")
    )
    args.dropout = args.dropout if args.dropout is not None else config.get("dropout")
    args.start_lr = (
        args.start_lr if args.start_lr is not None else config.get("start_lr")
    )
    args.max_audio_len = (
        args.max_audio_len
        if args.max_audio_len is not None
        else config.get("max_audio_len")
    )
    args.weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else config.get("weight_decay")
    )

    args.aux_lb_size = (
        args.aux_lb_size
        if args.aux_lb_size is not None
        else config.get("aux_lb_size")
    )

    return args


# modified from: https://github.com/ContinualAI/avalanche/blob/e91183e6fee8ccfdcde79f674e0c427f72bb9e8c/avalanche/training/plugins/early_stopping.py
class EarlyStoppingPlugin(SupervisedPlugin):
    """Early stopping and model checkpoint plugin.
    The plugin checks a metric and stops the training loop when the accuracy
    on the metric stopped progressing for `patience` epochs.
    After training, the best model's checkpoint is loaded.
    .. warning::
        The plugin checks the metric value, which is updated by the strategy
        during the evaluation. This means that you must ensure that the
        evaluation is called frequently enough during the training loop.
        For example, if you set `patience=1`, you must also set `eval_every=1`
        in the `BaseStrategy`, otherwise the metric won't be updated after
        every epoch/iteration. Similarly, `peval_mode` must have the same
        value.
    """

    def __init__(
        self,
        patience: int,
        val_stream_name: str,
        metric_name: str = "Top1_Acc_Stream",
        mode: str = "max",
        peval_mode: str = "epoch",
    ):
        """Init.
        :param patience: Number of epochs to wait before stopping the training.
        :param val_stream_name: Name of the validation stream to search in the
        metrics. The corresponding stream will be used to keep track of the
        evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
        reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
        given metric should me maximized (resp. minimized).
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            early stopping should happen after `patience`
            epochs or iterations (Default='epoch').
        """
        super().__init__()
        self.val_stream_name = val_stream_name
        self.patience = patience

        assert peval_mode in {"epoch", "iteration"}
        self.peval_mode = peval_mode

        self.metric_name = metric_name
        self.metric_key = f"{self.metric_name}/eval_phase/" f"{self.val_stream_name}"
        print(self.metric_key)
        if mode not in ("max", "min"):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == "max" else operator.lt

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step = None

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None

    def before_training_iteration(self, strategy, **kwargs):
        if self.peval_mode == "iteration":
            self._update_best(strategy)
            curr_step = self._get_strategy_counter(strategy)
            if curr_step - self.best_step >= self.patience:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()

    def before_training_epoch(self, strategy, **kwargs):
        if self.peval_mode == "epoch":
            self._update_best(strategy)
            curr_step = self._get_strategy_counter(strategy)
            print(f"Metric name: {self.metric_name}")
            print(f"Metric key: {self.metric_key}")
            res = strategy.evaluator.get_last_metrics()
            full_name = [k for k in res.keys() if k.startswith(self.metric_key)][-1]
            print(f"Full name of current metric: {full_name}")
            print(
                f"Current step: {curr_step}, current val metric: {strategy.evaluator.get_last_metrics().get(full_name)}"
            )
            print(f"Best step: {self.best_step}, best metric: {self.best_val}")
            print(f"Patience: {self.patience}")
            if curr_step - self.best_step >= self.patience:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        """
        since as we consider new experiences, the full name of the metric will contain the name of the experience
        we access the newest metric that starts with the key we monitoring 
        In the current example, during the training on the first experience, the retrieved value would be
        "Loss_Exp/eval_phase/valid_stream/Task000/Exp000", while for the second experience it would be
        "Loss_Exp/eval_phase/valid_stream/Task000/Exp001"
        """
        full_name = [k for k in res.keys() if k.startswith(self.metric_key)][-1]
        val_acc = res.get(full_name)
        if self.best_val is None:
            warnings.warn(
                f"Metric {self.metric_name} used by the EarlyStopping plugin "
                f"is not computed yet. EarlyStopping will not be triggered."
            )
        if self.best_val is None or self.operator(val_acc, self.best_val):
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            self.best_step = self._get_strategy_counter(strategy)

    def _get_strategy_counter(self, strategy):
        if self.peval_mode == "epoch":
            return strategy.clock.train_exp_epochs
        elif self.peval_mode == "iteration":
            return strategy.clock.train_exp_iterations
        else:
            raise ValueError("Invalid `peval_mode`:", self.peval_mode)
