import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, temperature=None):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]).cuda()) if temperature is not None else (
            nn.Parameter(torch.tensor([1.0]).cuda()))

    def train(self, logits, labels):
        return 0.0, 0.0

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=-1)
            return softmax(logits / self.temperature)

        return logits / self.temperature


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


class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([1.5]).cuda())

    def train(self, logits, labels, softmax=True):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        ece_before = ece_criterion(logits, labels)
        print("ece_before: %.4f" % ece_before.item())

        optimizer = optim.SGD([self.temperature], lr=0.1)
        for iter in range(50):
            optimizer.zero_grad()
            logits = logits.cuda()
            logits.requires_grad = True
            out = logits / self.temperature
            loss = nll_criterion(out, labels.long().cuda())
            loss.backward()
            optimizer.step()

        print('Optimal temperature: %.3f' % self.temperature.item())
        out = logits / self.temperature
        ece_after = ece_criterion(out, labels)
        print("ece_after: %.4f" % ece_after.item())

        return ece_before.item(), ece_after.item()

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits / self.temperature)

        return logits / self.temperature


class PlattScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.5]).cuda())
        self.b = nn.Parameter(torch.tensor([1.5]).cuda())

    def train(self, logits, labels, softmax=True):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        ece_before = ece_criterion(logits, labels)
        print("ece_before: %.4f" % ece_before.item())

        optimizer = optim.LBFGS([self.a, self.b], lr=0.1, max_iter=100)

        def eval():
            optimizer.zero_grad()
            out = logits * self.a + self.b
            loss = nll_criterion(out, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        out = logits * self.a + self.b
        ece_after = ece_criterion(out, labels)
        print("ece_after: %.4f" % ece_after.item())

        return ece_before.item(), ece_after.item()

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits * self.a + self.b)

        return logits * self.a + self.b


class VectorScaling(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 1000
        self.w = nn.Parameter((torch.ones(num_classes) * 1.5).cuda())
        self.b = nn.Parameter((torch.rand(num_classes) * 2.0 - 1.0).cuda())

    def train(self, logits, labels, softmax=True):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        ece_before = ece_criterion(logits, labels)
        print("ece_before: %.4f" % ece_before.item())

        optimizer = optim.LBFGS([self.w, self.b], lr=0.05, max_iter=100)

        def eval():
            optimizer.zero_grad()
            out = logits * self.w + self.b
            loss = nll_criterion(out, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        out = logits * self.w + self.b
        ece_after = ece_criterion(out, labels)
        print("ece_after: %.4f" % ece_after.item())

        return ece_before.item(), ece_after.item()

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits * self.w + self.b)

        return logits * self.w + self.b
