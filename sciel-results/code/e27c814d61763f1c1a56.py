"""Improved greedy set-cover for C(v,k,t): each block maximizes newly-covered t-subsets."""
import time
import numpy as np

_T2I = {}
_INV = {}


def _tables(v, t):
    key = (v, t)
    if key not in _T2I:
        d = {}
        inv = []
        for i, c in enumerate(np.ndindex(*([v] * t))):
            s = tuple(sorted(c))
            d[s] = i
            inv.append(s)
        _T2I[key] = d
        _INV[key] = inv
    return _T2I[key], _INV[key]


def _tset(block, idx):
    n = len(block)
    t = len(idx[0]) if idx else 0
    out = []
    for i in range(n - t + 1):
        a = block[i:i + t]
        out.append(idx[tuple(sorted(a))])
    return out


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0

    idx, inv = _tables(v, t)
    nT = len(idx)
    covered = np.zeros(nT, dtype=bool)
    blocks = []
    pts = list(range(v))

    def timeup():
        return time.time() > deadline

    while True:
        if timeup():
            break
        if covered.all():
            break

        block = []
        while len(block) < k:
            if timeup():
                break
            cands = [x for x in pts if x not in block]
            if not cands:
                break
            best_x = None
            best_gain = -1
            for x in cands:
                if timeup():
                    break
                nb = block + [x]
                tsets = _tset(nb, idx)
                gain = int((~covered[tsets]).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                best_x = cands[0]
            block.append(best_x)

        if len(block) < k:
            break
        block.sort()
        tsets = _tset(block, idx)
        newmask = ~covered[tsets]
        if newmask.any():
            covered[tsets] = covered[tsets] | newmask
            blocks.append(block)
        else:
            break

    return blocks
