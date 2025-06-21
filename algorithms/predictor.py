import warnings
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from commons.utils import split_logits_labels, get_device, build_score
from .metric import Metrics


class Predictor(nn.Module):
    def __init__(self, model, preprocessor, conformal, alpha, random=True, penalty=0.001):
        super(Predictor, self).__init__()
        self._model = model
        self.score_function = build_score(conformal, penalty=penalty)
        self._preprocessor = preprocessor
        self.alpha = alpha
        self.num_classes = 1000
        self._metric = Metrics()
        self._device = get_device()
        self.random = random

    def calibrate(self, calib_calibloader, conf_calibloader):
        calib_logits, calib_labels = split_logits_labels(self._model, calib_calibloader)
        ece_before, ece_after = self._preprocessor.train(calib_logits, calib_labels)
        conf_logits, conf_labels = split_logits_labels(self._model, conf_calibloader)
        conf_logits = self._preprocessor(conf_logits, softmax=False)
        self.calculate_threshold(conf_logits, conf_labels)
        return ece_before, ece_after

    def calibrate_with_logits_labels(self, logits, labels):
        ece_before, ece_after = self._preprocessor.train(logits, labels)
        self.ece_before = ece_before
        self.ece_after = ece_after
        logits = self._preprocessor(logits, softmax=False)
        self.calculate_threshold(logits, labels)

    def calculate_threshold(self, logits, labels):
        alpha = self.alpha
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels, random=self.random)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        if len(scores) == 0:
            warnings.warn(
                "The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is "
                "set as torch.inf.")
            return torch.inf
        qunatile_value = math.ceil(scores.shape[0] + 1) * (1 - alpha) / scores.shape[0]

        if qunatile_value > 1:
            warnings.warn(
                "The value of quantile exceeds 1. It should be a value in (0,1). To avoid program crash, the threshold "
                "is set as torch.inf.")
            return torch.inf

        return torch.quantile(scores, qunatile_value, interpolation="higher").to(self._device)

    def predict(self, x_batch):
        self._model.eval()
        if self._model is not None:
            tmp_logits = self._model(x_batch.to(self._device)).float()
        tmp_logits = self._preprocessor(tmp_logits, softmax=False).detach()
        sets, scores = self.predict_with_logits(tmp_logits)
        return sets, scores

    def predict_with_logits(self, logits, q_hat=None):
        scores = self.score_function(logits, random=self.random).to(self._device)
        if q_hat is None:
            S = self._generate_prediction_set(scores, self.q_hat)
        else:
            S = self._generate_prediction_set(scores, q_hat)
        return S, scores

    def evaluate(self, val_dataloader):
        prediction_sets = []
        probs_list = []
        labels_list = []
        scores_list = []
        with torch.no_grad():
            for examples in tqdm(val_dataloader):
                tmp_x, tmp_label = examples[0].to(self._device), examples[1].to(self._device)
                prediction_sets_batch, scores_batch = self.predict(tmp_x)
                target_scores_batch = scores_batch[range(tmp_label.shape[0]), tmp_label]
                prediction_sets.extend(prediction_sets_batch)
                tmp_probs = self._preprocessor(self._model(tmp_x)).detach()
                probs_list.append(tmp_probs)
                labels_list.append(tmp_label)
                scores_list.append(target_scores_batch)
        val_probs = torch.cat(probs_list)
        val_labels = torch.cat(labels_list)
        val_scores = torch.cat(scores_list)
        res_dict = {"top1": self._metric('accuracy')(val_probs, val_labels, [1]),
                    "top5": self._metric('accuracy')(val_probs, val_labels, [5]),
                    "Coverage_rate": self._metric('coverage_rate')(prediction_sets, val_labels),
                    "Average_size": self._metric('average_size')(prediction_sets, val_labels),
                    }
        return res_dict

    def _generate_prediction_set(self, scores, q_hat):
        if len(scores.shape) == 1:
            return torch.argwhere(scores <= q_hat).reshape(-1).tolist()
        else:
            return [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
