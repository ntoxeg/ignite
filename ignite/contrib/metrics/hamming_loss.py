from functools import partial

import torch
try:
    from sklearn.metrics import hamming_loss
except ImportError:
    raise RuntimeError("This contrib module requires scikit-learn to be installed.")

from ignite.metrics import EpochMetric


def hamming_loss_compute_fn(y_preds, y_targets, activation=None, threshold=0.5):
    y_targets = y_targets.numpy()
    if activation is not None:
        y_preds = activation(y_preds)
    y_preds = torch.gt(y_preds, threshold).numpy()
    return hamming_loss(y_targets, y_preds)


class HammingLoss(EpochMetric):
    """Computes the Hamming Loss metric over predictions and labels gathered during an epoch

    Uses `sklearn.metrics.hamming_loss <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss>`.

    Args:
        activation (Callable, optional): a function to apply on predictions,
            e.g. `activation=torch.sigmoid` when the model returns logits.
        threshold (float, optional): when deciding wheather a prediction for a given label
            is positive or negative this will be the threshold value.
        output_transform (Callable, optional): a callable that is used to transform
            the :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model
            and you want to compute the metric with respect to one of the outputs.
    """
    def __init__(self, activation=None, threshold=0.5, output_transform=lambda x: x):
        super().__init__(partial(hamming_loss_compute_fn, activation=activation, threshold=threshold),
                         output_transform=output_transform)

