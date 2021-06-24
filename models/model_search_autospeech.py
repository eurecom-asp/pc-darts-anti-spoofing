import math
import torch.nn.functional as F
from func.operations import *
from utils.utils import Genotype, gumbel_softmax, drop_path
import torchaudio
from _collections import OrderedDict
from func.frontend import LFCC, LFB, Spectrogram


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

primitives_2 = OrderedDict([('primitives_normal', 14 * [PRIMITIVES]),
                            ('primitives_reduct', 14 * [PRIMITIVES])])

class MixedOp(nn.Module):

    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        This is a forward function.
        :param x: Feature map
        :param weights: A tensor of weight controlling the path flow
        :return: A weighted sum of several path
        """
        output = 0
        for op_idx, op in enumerate(self._ops):
            if weights[op_idx].item() != 0:
                if math.isnan(weights[op_idx]):
                    raise OverflowError(f'weight: {weights}')
            output += weights[op_idx] * op(x)
        return output


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
            print('FactorizedReduce')
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            print('ReLUConvBN')
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights, drop_prob=0.0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        offset = 0
        
        for i in range(self._steps):
            if drop_prob > 0. and self.training:
                s = sum(
                    drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            # s = sum(weights2[offset+j].to(self._ops[offset+j](h, weights[offset+j]).device)*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        out = torch.cat(states[-self._multiplier:], dim=1)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, layers, args, num_classes, criterion, front_end, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = 0

        nn.Module.PRIMITIVES = primitives_2

        C_curr = stem_multiplier * C
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        
        self._initialize_alphas()

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

        self.classifier = nn.Linear(C_prev, self._num_classes)

    def new(self):
        model_new = Network(self._C, self._embed_dim, self._layers, self._criterion,
                            self.PRIMITIVES, drop_path_prob=self.drop_path_prob).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, is_mask=False, discrete=False):
        input = self.feature(input, is_mask)
        input = input.permute(0, 2, 1).unsqueeze(1)
        s0 = s1 = self.stem(input)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if discrete:
                    weights = self.alphas_reduce
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_reduce, dim=-1))
            else:
                if discrete:
                    weights = self.alphas_normal
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_normal, dim=-1))
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
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

    def _loss(self, input, target, is_mask=False):
        logits, _ = self(input, is_mask)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def compute_arch_entropy(self, dim=-1):
        alpha = self.arch_parameters()[0]
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError:  # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[start+j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

