from functools import partial

import torch
from ignite.metrics import EpochMetric


def _compute_fn(y_preds, y_targets, metric_fn, activation=None):
    y_targets = y_targets.numpy()
    if activation is not None:
        y_preds = activation(y_preds)
    y_preds = y_preds.numpy()
    return metric_fn(y_preds, y_targets)


class EpochLambda(EpochMetric):
    """Computes a metric over predictions and labels gathered during an epoch

    Args:
        metric_fn (Callable): a function for computing your metric,
            it must take arguments of `(y_preds, y_targets)`,
            where `y_preds` is your model's predictions and `y_targets`
            are ground-truth values.
        activation (Callable, optional): a function to apply on predictions,
            e.g. `activation=torch.sigmoid` when the model returns logits.
        output_transform (Callable, optional): a callable that is used to transform
            the :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model
            and you want to compute the metric with respect to one of the outputs.
    """
    def __init__(self, metric_fn, activation=None, output_transform=lambda x: x):
        super().__init__(partial(_compute_fn, metric_fn=metric_fn, activation=activation),
                         output_transform=output_transform)

