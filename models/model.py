from func.operations import *
from func.frontend import LFCC, LFB, Spectrogram

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class Network(nn.Module):

  def __init__(self, C, layers, args, num_classes, genotype, front_end):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers


    self.stem0 = nn.Sequential(
      nn.Conv2d(1, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      if reduction:
        print('Reduction Cell')
      else:
        print('Normal Cell')
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    if front_end == 'Spectrogram':
      self.feature = Spectrogram(args.nfft, args.nfft//args.hop, args.nfft, args.sr, args.is_log, args.is_cmvn)
      print('*****Using Spectrogram front end*****')
    elif front_end == 'LFCC':
      self.feature = LFCC(args.nfft, args.nfft//args.hop, args.nfft, args.sr, args.nfilter, args.num_ceps, args.is_cmvn)
      print('*****Using LPCC front end*****') 
    elif front_end == 'LFB':
      self.feature = LFB(args.nfft, args.nfft//args.hop, args.nfft, args.sr, args.nfilter, args.num_ceps, args.is_cmvn)
      print('*****Using LFB front end*****')
    else:
      print('*****NO front end selected*****')

    self.classifier = nn.Linear(C_prev, num_classes)


  def forward(self, input, is_mask=False):
    input = self.feature(input, is_mask)
    # transpose input shape from (batch, time, freq) to (batch, channel=1, freq, time)
    input = input.permute(0, 2, 1).unsqueeze(1)
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    v = self.global_pooling(s1)
    embeddings = v.view(v.size(0), -1)
    if not self.training:
      return embeddings
    else:
      logits = self.classifier(embeddings)
      return logits, embeddings
  
  def forward_classifier(self, embeddings):
    logits = self.classifier(embeddings)
    return logits




