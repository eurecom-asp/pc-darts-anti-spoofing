import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import roc_curve
from collections import namedtuple
from prettytable import PrettyTable

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6}")
    return total_params/1e6


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
      
class EERMeter(object):
    """
    Class to gradually store network scores and return EER on demand.
    EER is computed lazily (since you can't really do it otherwise).
    It internally stores predictions and true labels, so it can get quite big. I'm sorry.
    Must be fed raw activations from the network. Applies a softmax to turn them into scores.

    Params:
      name: some name that will be printed when calling str()
      round_digits: how many digits to round the eer to (only when printing)
      positive_label: which label is the one to consider the "positive" class. Can either be 0 or 1
    """

    def __init__(self, name, round_digits=4, positive_label=1, percent=True):
        assert positive_label in [0, 1], "Positive label must be either 0 or 1"
        self.name = name
        self.round_digits = round_digits
        self.positive_label = positive_label
        self.reset()
        self.percent = percent

    def reset(self):
        """
        Empty internal array values.
        """
        self.y_true = []
        self.y_score = []

    def update(self, new_true, new_preds):
        """
        Add network outputs to the track of scores.
        new_preds must be raw network activations, while new_preds are ground_truth labels.
        """
        new_score = F.softmax(new_preds, dim=1)
        self.y_true += new_true.tolist()
        self.y_score += new_score[:, self.positive_label].tolist() # Extract only the column associated to the desired label

    def get_eer(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_score, pos_label=self.positive_label)
        eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
        return eer

    def __str__(self):
        """
        Displays EER obtained so far and the number of elements used to compute it.
        """
        eer = self.get_eer()
        if self.percent:
            mult = 100
            symbol = '%'
        else:
            mult = 1
            symbol = ''
        fmtstr = f' | {self.name}: {round(eer*mult, self.round_digits)}{symbol} | Num. Elements: {len(self.y_true)}'
        return fmtstr


class TotalAccuracyMeter(object):
    """
    Keeps track of correct predictions out of the total and computes total accuracy over all predictions made so far.

    Params:
        name: some name that will be displayed when printing
        round_digits: rounding of the accuracy decimal digits (only used when printing)
    """

    def __init__(self, name, round_digits=4, percent=True):
        self.name = name
        self.round_digits = round_digits
        self.percent = percent
        self.reset()

    def reset(self):
        """
        Reset all counts.
        """
        self.correct = 0
        self.total = 0

    def update(self, new_target, new_activations):
        """
        Update the correct prediction count and the total count.
        """
        new_preds = new_activations.argmax(dim=1)
        self.correct += (new_preds == new_target).sum().item()
        self.total += len(new_target)

    def get_accuracy(self):
        return self.correct/self.total

    def __str__(self):
        acc = self.get_accuracy()
        if self.percent:
            mult = 100
            symbol = '%'
        else:
            mult = 1
            symbol = ''
        fmtstr = f' | {self.name}: {round(acc*mult, self.round_digits)}{symbol} | Num. Elements: {self.total}'
        return fmtstr


def gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    """
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if torch.isnan(ret).sum():
        import ipdb
        ipdb.set_trace()
        raise OverflowError(f'gumbel softmax output: {ret}')
    return ret