import torch


class aps(object):
    def __call__(self, logits, label=None, random=True):
        assert len(logits.shape) <= 2, "The dimension of logits must be less than 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if label is None:
            return self._calculate_all_label(probs, random=random)
        else:
            return self._calculate_single_label(probs, label, random=random)

    def _calculate_all_label(self, probs, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        if random:
            U = torch.rand(probs.shape, device=probs.device)
            ordered_scores = cumsum - ordered * U
        else:
            ordered_scores = cumsum
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = torch.where(indices == label.view(-1, 1))
        idx_minus_one = (idx[0], idx[1] - 1)
        if random:
            U = torch.rand(indices.shape[0], device=probs.device)
            scores_first_rank = U * cumsum[idx]
            scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
            scores = torch.where(idx[1] == 0, scores_first_rank, scores_usual)
        else:
            scores = cumsum[range(cumsum.shape[0]), label]
        return scores

class raps(aps):
    """
    Regularized Adaptive Prediction Sets (Angelopoulos et al., 2020)
    paper : https://arxiv.org/abs/2009.14193

    :param penalty: the weight of regularization. When penalty = 0, RAPS=APS.
    :param kreg: the rank of regularization which is an integer in [0,labels_num].
    """

    def __init__(self, penalty=0.001, kreg=0, random=True):

        if penalty <= 0:
            raise ValueError("The parameter 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("The parameter 'kreg' must be a nonnegative value.")
        if type(kreg) != int:
            raise TypeError("The parameter 'kreg' must be a integer.")
        super(raps, self).__init__()
        if penalty is None:
            penalty = 0.001
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label, random=True):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores_first_rank = U * ordered[idx] + reg
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
