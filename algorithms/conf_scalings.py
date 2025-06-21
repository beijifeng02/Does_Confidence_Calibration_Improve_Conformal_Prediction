import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .predictor import Predictor
from .scalings import Identity
from .predictor import Predictor


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    The input to this loss is the logits of a model, NOT the softmax scores.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, softmax=True):
        if softmax:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits

        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class ConformalTemperatureScaling(nn.Module):
    def __init__(self, model, alpha):
        super().__init__()
        self.alpha = alpha
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.5)).cuda())
        preprocessor = Identity(temperature=1.0)
        self.predictor = Predictor(preprocessor=preprocessor, conformal="aps", model=model, alpha=self.alpha)
        self.lr = 0.8
        self.stop = 0.05

    def train(self, logits, labels):
        ece_criterion = _ECELoss()
        optimizer = optim.SGD([self.temperature], lr=self.lr)
        ece_before = ece_criterion(logits, labels)

        for iter in range(10000):
            optimizer.zero_grad()
            T_old = self.temperature.item()
            out = logits / torch.exp(self.temperature)
            loss = self.criterion(out, labels)
            loss.backward()
            optimizer.step()

            if abs(self.temperature.item() - T_old) < self.stop:
                break

        out = logits / self.temperature
        ece_after = ece_criterion(out, labels)
        return ece_before, ece_after

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits / torch.exp(self.temperature))

        return logits / torch.exp(self.temperature)

    def criterion(self, logits, labels, fraction=0.5):
        val_split = int(fraction * logits.shape[0])
        cal_logits = logits[:val_split]
        cal_labels = labels[:val_split]
        test_logits = logits[val_split:]
        test_labels = labels[val_split:]

        self.predictor.calculate_threshold(cal_logits, cal_labels, random=False)
        tau = self.predictor.q_hat
        test_scores = self.predictor.score_function(test_logits, random=False)
        loss = torch.mean((tau - test_scores[range(test_scores.shape[0]), test_labels]) ** 2)
        return loss


class ConformalPlattScaling(nn.Module):
    def __init__(self, model, alpha):
        super().__init__()
        self.alpha = alpha
        num_classes = 1000
        self.a = nn.Parameter(torch.tensor([1.5]).cuda())
        self.b = nn.Parameter(torch.tensor([1.5]).cuda())
        preprocessor = Identity(temperature=1.0)
        self.predictor = Predictor(preprocessor=preprocessor, conformal="aps", model=model, alpha=self.alpha)
        self.lr = 0.5
        self.stop = 0.05

    def train(self, logits, labels):
        ece_criterion = _ECELoss()
        ece_before = ece_criterion(logits, labels)
        optimizer = optim.SGD([self.a, self.b], lr=self.lr)

        for iter in range(1000):
            optimizer.zero_grad()
            out = logits * self.a + self.b
            loss = self.criterion(out, labels)
            loss.backward()
            optimizer.step()

        # print('Optimal temperature: %.3f' % torch.exp(self.temperature).item())
        out = logits * self.a + self.b
        ece_after = ece_criterion(out, labels)
        print(f"ECE before:{ece_before}")
        print(f"ECE after:{ece_after}")
        return 0.0, 0.0

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits * self.a + self.b)

        return logits * self.a + self.b

    def criterion(self, logits, labels, fraction=0.5):
        val_split = int(fraction * logits.shape[0])
        cal_logits = logits[:val_split]
        cal_labels = labels[:val_split]
        test_logits = logits[val_split:]
        test_labels = labels[val_split:]

        self.predictor.calculate_threshold(cal_logits, cal_labels, random=False)
        tau = self.predictor.q_hat
        test_scores = self.predictor.score_function(test_logits, random=False)
        loss = torch.mean((tau - test_scores[range(test_scores.shape[0]), test_labels]) ** 2)
        return loss


class ConformalVectorScaling(nn.Module):
    def __init__(self, model, alpha):
        super().__init__()
        self.alpha = alpha
        num_classes = 1000
        self.w = nn.Parameter((torch.ones(num_classes) * 1.5).cuda())
        self.b = nn.Parameter((torch.rand(num_classes) * 2.0 - 1.0).cuda())
        preprocessor = Identity(temperature=1.0)
        self.predictor = Predictor(preprocessor=preprocessor, model=model, conformal="aps", alpha=self.alpha)
        self.lr = 0.8
        self.stop = 0.05

    def train(self, logits, labels):
        optimizer = optim.SGD([self.w, self.b], lr=self.lr)

        for iter in range(100):
            optimizer.zero_grad()
            out = logits * self.w + self.b
            loss = self.criterion(out, labels)
            loss.backward()
            optimizer.step()

        # print('Optimal temperature: %.3f' % torch.exp(self.temperature).item())
        return 0.0, 0.0

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits * self.w + self.b)

        return logits * self.w + self.b

    def criterion(self, logits, labels, fraction=0.5):
        val_split = int(fraction * logits.shape[0])
        cal_logits = logits[:val_split]
        cal_labels = labels[:val_split]
        test_logits = logits[val_split:]
        test_labels = labels[val_split:]

        self.predictor.calculate_threshold(cal_logits, cal_labels, random=False)
        tau = self.predictor.q_hat
        test_scores = self.predictor.score_function(test_logits, random=False)
        loss = torch.mean((tau - test_scores[range(test_scores.shape[0]), test_labels]) ** 2)
        return loss
