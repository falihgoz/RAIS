import numpy as np
import sys
import torch
from torch import Tensor
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from collections import defaultdict
from typing import List, Optional, Union, Dict

# Python implementation of EER, min-tDCF


class EER(Metric[float]):
    """Accuracy metric. This is a standalone metric.

    The update method computes the accuracy incrementally
    by keeping a running average of the <prediction, target> pairs
    of Tensors provided over time.

    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self):
        """Creates an instance of the standalone Accuracy metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        self.all_scores = []  # Empty tensor for scores
        self.all_labels = []  # Empty tensor for labels
        """The mean utility that will be used to store the running accuracy."""

    @torch.no_grad()
    def update(
        self,
        scores: Tensor,
        true_y: Tensor,
    ) -> None:
        """Update the running accuracy given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.

        :return: None.
        """
        scores = scores.data.cpu().numpy()
        true_y = true_y.data.cpu().numpy()

        self.all_scores.append(scores)
        self.all_labels.append(true_y)

    def result(self) -> float:
        """Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The current running accuracy, which is a float value
            between 0 and 1.
        """
        self.all_scores = np.concatenate(self.all_scores, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)

        eer, _ = compute_eer_alternative(-self.all_scores[:, 0], self.all_labels)
        # other_eer, _ = compute_eer_alternative(self.all_scores[:, 1], self.all_labels)
        # print(eer)
        # print(other_eer)
        # eer = compute_eer(
        #     -self.all_scores[self.all_labels == 0, 0],
        #     -self.all_scores[self.all_labels == 1, 0],
        # )[0]
        # other_eer = compute_eer(
        #     self.all_scores[self.all_labels == 0, 1],
        #     self.all_scores[self.all_labels == 1, 1],
        # )[0]
        # print(eer)
        # print(other_eer)
        # eer_cm = min(eer, other_eer)
        # return eer_cm
        return eer

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self.all_scores = []  # Empty tensor for scores
        self.all_labels = []  # Empty tensor for labels


class EERPluginMetric(GenericPluginMetric[float, EER]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(EER(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class MinibatchEER(EERPluginMetric):
    """
    The minibatch plugin EER metric.
    This metric only works at training time.

    This metric computes the average EER over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochEER` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchEER metric.
        """
        super(MinibatchEER, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_EER_MB"


class EpochEER(EERPluginMetric):
    """
    The average EER over a single training epoch.
    This plugin metric only works at training time.

    The EER will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochEER metric.
        """

        super(EpochEER, self).__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "Top1_EER_Epoch"


class RunningEpochEER(EERPluginMetric):
    """
    The average EER across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the EER averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochEER metric.
        """

        super(RunningEpochEER, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_RunningEER_Epoch"


class ExperienceEER(EERPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average EER over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceEER metric
        """
        super(ExperienceEER, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Top1_EER_Exp"


class StreamEER(EERPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average EER over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamEER metric
        """
        super(StreamEER, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Top1_EER_Stream"


class TrainedExperienceEER(EERPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    EER for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceEER metric by first
        constructing EERPluginMetric
        """
        super(TrainedExperienceEER, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )
        self._current_experience = 0

    def after_training_exp(self, strategy):
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        self.reset()
        return super().after_training_exp(strategy)

    def update(self, strategy):
        """
        Only update the EER with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            EERPluginMetric.update(self, strategy)

    def __str__(self):
        return "EER_On_Trained_Experiences"


def EER_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[EERPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch EER at training time.
    :param epoch: If True, will return a metric able to log
        the epoch EER at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch EER at training time.
    :param experience: If True, will return a metric able to log
        the EER on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the EER averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation EER only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics: List[EERPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchEER())

    if epoch:
        metrics.append(EpochEER())

    if epoch_running:
        metrics.append(RunningEpochEER())

    if experience:
        metrics.append(ExperienceEER())

    if stream:
        metrics.append(StreamEER())

    if trained_experience:
        metrics.append(TrainedExperienceEER())

    return metrics


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size)
    )  # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_eer_alternative(scores, labels):
    if isinstance(scores, list) is False:
        scores = list(scores)
    if isinstance(labels, list) is False:
        labels = list(labels)

    target_scores = []
    nontarget_scores = []

    for item in zip(scores, labels):
        if item[1] == 1:
            target_scores.append(item[0])
        else:
            nontarget_scores.append(item[0])

    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    if target_size == 0 or nontarget_size == 0:
        print("Target size: ", target_size)
        print("non-target size: ", nontarget_size)
        raise ValueError("Target or non-target scores are empty.")

    target_position = 0
    for i in range(target_size - 1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break

    if target_position >= len(target_scores):
        raise IndexError(
            f"Target position {target_position} is out of bounds for target_scores of size {len(target_scores)}."
        )

    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size
    return eer, th


def compute_tDCF(
    bonafide_score_cm,
    spoof_score_cm,
    Pfa_asv,
    Pmiss_asv,
    Pmiss_spoof_asv,
    cost_model,
    print_cost,
):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
    """

    # Sanity check of cost parameters
    if (
        cost_model["Cfa_asv"] < 0
        or cost_model["Cmiss_asv"] < 0
        or cost_model["Cfa_cm"] < 0
        or cost_model["Cmiss_cm"] < 0
    ):
        print("WARNING: Usually the cost values should be positive!")

    if (
        cost_model["Ptar"] < 0
        or cost_model["Pnon"] < 0
        or cost_model["Pspoof"] < 0
        or np.abs(cost_model["Ptar"] + cost_model["Pnon"] + cost_model["Pspoof"] - 1)
        > 1e-10
    ):
        sys.exit(
            "ERROR: Your prior probabilities should be positive and sum up to one."
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            "ERROR: you should provide miss rate of spoof tests against your ASV system."
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit("ERROR: Your scores contain nan or inf.")

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit("ERROR: You should provide soft CM scores - not binary decisions")

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm
    )

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = (
        cost_model["Ptar"]
        * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * Pmiss_asv)
        - cost_model["Pnon"] * cost_model["Cfa_asv"] * Pfa_asv
    )
    C2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            "You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?"
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print(
            "t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n".format(
                bonafide_score_cm.size, spoof_score_cm.size
            )
        )
        print("t-DCF MODEL")
        print(
            "   Ptar         = {:8.5f} (Prior probability of target user)".format(
                cost_model["Ptar"]
            )
        )
        print(
            "   Pnon         = {:8.5f} (Prior probability of nontarget user)".format(
                cost_model["Pnon"]
            )
        )
        print(
            "   Pspoof       = {:8.5f} (Prior probability of spoofing attack)".format(
                cost_model["Pspoof"]
            )
        )
        print(
            "   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)".format(
                cost_model["Cfa_asv"]
            )
        )
        print(
            "   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)".format(
                cost_model["Cmiss_asv"]
            )
        )
        print(
            "   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)".format(
                cost_model["Cfa_cm"]
            )
        )
        print(
            "   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)".format(
                cost_model["Cmiss_cm"]
            )
        )
        print(
            "\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)"
        )

        if C2 == np.minimum(C1, C2):
            print(
                "   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n".format(C1 / C2)
            )
        else:
            print(
                "   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n".format(C2 / C1)
            )

    return tDCF_norm, CM_thresholds


__all__ = [
    "EER_metrics",
]
