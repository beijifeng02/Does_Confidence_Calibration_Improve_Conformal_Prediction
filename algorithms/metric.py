import numpy as np
import torch

from typing import Any


__all__ = ["Registry"]


class Registry:
    """A registry providing name -> object mapping, to support
    custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone(nn.Module):
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        if name in self._obj_map and not force:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )

        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        if obj is None:
            # Used as a decorator
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class

            return wrapper

        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj, force=force)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )

        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())

METRICS_REGISTRY = Registry("METRICS")


@METRICS_REGISTRY.register()
def coverage_rate(prediction_sets, labels):
    cvg = 0
    for index, ele in enumerate(zip(prediction_sets, labels)):
        if ele[1] in ele[0]:
            cvg += 1
    return cvg / len(prediction_sets)


@METRICS_REGISTRY.register()
def average_size(prediction_sets, labels):
    avg_size = 0
    for index, ele in enumerate(prediction_sets):
        avg_size += len(ele)
    return avg_size / len(prediction_sets)


@METRICS_REGISTRY.register()
def accuracy(probs, targets, top_k=(1,)):
    k_max = max(top_k)
    batch_size = targets.size(0)

    _, order = probs.topk(k_max, dim=1, largest=True, sorted=True)
    order = order.t()
    correct = order.eq(targets.view(1, -1).expand_as(order))

    acc = []
    for k in top_k:
        correct_k = correct[:k].float().sum()
        acc.append(correct_k.mul_(100.0 / batch_size))
    return acc[0].item()


@METRICS_REGISTRY.register()
def adaptiveness(prediction_sets, probs, labels):
    sort_probs = torch.argsort(probs, descending=True)
    rank = (sort_probs == labels.view(-1, 1)).nonzero()[:, 1] + 1
    result = {}

    # top 1
    index = torch.where(rank == 1)[0]
    tmp_prediction_sets = [prediction_sets[i] for i in index]
    tmp_labels = [labels[i] for i in index]
    cvg = 0;
    avg_size = 0
    for index, ele in enumerate(zip(tmp_prediction_sets, tmp_labels)):
        avg_size += len(ele[0])
        if ele[1] in ele[0]:
            cvg += 1
    avg_size /= len(tmp_prediction_sets)
    cvg /= len(tmp_prediction_sets)
    result["1"] = {"cnt": len(tmp_prediction_sets),
                   "coverage": cvg,
                   "size": avg_size}

    # top 2-3
    index = torch.where((rank >= 2) & (rank <= 3))[0]
    tmp_prediction_sets = [prediction_sets[i] for i in index]
    tmp_labels = [labels[i] for i in index]
    cvg = 0;
    avg_size = 0
    for index, ele in enumerate(zip(tmp_prediction_sets, tmp_labels)):
        avg_size += len(ele[0])
        if ele[1] in ele[0]:
            cvg += 1
    avg_size /= len(tmp_prediction_sets)
    cvg /= len(tmp_prediction_sets)
    result["2-3"] = {"cnt": len(tmp_prediction_sets),
                     "coverage": cvg,
                     "size": avg_size}

    # top 4-6
    index = torch.where((rank >= 4) & (rank <= 6))[0]
    tmp_prediction_sets = [prediction_sets[i] for i in index]
    tmp_labels = [labels[i] for i in index]
    cvg = 0;
    avg_size = 0
    for index, ele in enumerate(zip(tmp_prediction_sets, tmp_labels)):
        avg_size += len(ele[0])
        if ele[1] in ele[0]:
            cvg += 1
    avg_size /= len(tmp_prediction_sets)
    cvg /= len(tmp_prediction_sets)
    result["4-6"] = {"cnt": len(tmp_prediction_sets),
                     "coverage": cvg,
                     "size": avg_size}

    # top 7-10
    index = torch.where((rank >= 7) & (rank <= 10))[0]
    tmp_prediction_sets = [prediction_sets[i] for i in index]
    tmp_labels = [labels[i] for i in index]
    cvg = 0;
    avg_size = 0
    for index, ele in enumerate(zip(tmp_prediction_sets, tmp_labels)):
        avg_size += len(ele[0])
        if ele[1] in ele[0]:
            cvg += 1
    avg_size /= len(tmp_prediction_sets)
    cvg /= len(tmp_prediction_sets)
    result["7-10"] = {"cnt": len(tmp_prediction_sets),
                      "coverage": cvg,
                      "size": avg_size}

    # top 11-100
    index = torch.where((rank >= 11) & (rank <= 100))[0]
    tmp_prediction_sets = [prediction_sets[i] for i in index]
    tmp_labels = [labels[i] for i in index]
    cvg = 0;
    avg_size = 0
    for index, ele in enumerate(zip(tmp_prediction_sets, tmp_labels)):
        avg_size += len(ele[0])
        if ele[1] in ele[0]:
            cvg += 1
    avg_size /= len(tmp_prediction_sets)
    cvg /= len(tmp_prediction_sets)
    result["11-100"] = {"cnt": len(tmp_prediction_sets),
                        "coverage": cvg,
                        "size": avg_size}

    # top 101-1000
    index = torch.where((rank >= 101) & (rank <= 1000))[0]
    tmp_prediction_sets = [prediction_sets[i] for i in index]
    tmp_labels = [labels[i] for i in index]
    cvg = 0;
    avg_size = 0
    for index, ele in enumerate(zip(tmp_prediction_sets, tmp_labels)):
        avg_size += len(ele[0])
        if ele[1] in ele[0]:
            cvg += 1
    avg_size /= len(tmp_prediction_sets)
    cvg /= len(tmp_prediction_sets)
    result["101-1000"] = {"cnt": len(tmp_prediction_sets),
                          "coverage": cvg,
                          "size": avg_size}

    return result


@METRICS_REGISTRY.register()
def SSCV(prediction_sets, labels, alpha, stratified_size=[[0, 1], [2, 4], [5, 10], [10, 100], [101, 1000]]):
    """
    Size-stratified coverage violation (SSCV)
    """
    labels = labels.cpu()
    size_array = np.zeros(len(labels))
    correct_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        size_array[index] = len(ele)
        correct_array[index] = 1 if labels[index] in ele else 0

    sscv = []
    for stratum in stratified_size:
        temp_index = np.argwhere((size_array >= stratum[0]) & (size_array <= stratum[1]))
        if len(temp_index) > 0:
            stratum_violation = abs((1 - alpha) - np.mean(correct_array[temp_index]))
            sscv.append(stratum_violation)
    return np.mean(sscv)


class Metrics:

    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY.registered_names():
            raise NameError(f"The metric: {metric} is not defined in DeepCP.")
        return METRICS_REGISTRY.get(metric)


class DimensionError(Exception):
    pass
