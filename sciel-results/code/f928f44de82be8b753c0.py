import time
from itertools import combinations

P = {...}  # keep as is

def construct():
    triples = P['edge_order']
    n = P['n']
    full = (1 << n) - 1
    masks = []
    for tr in triples:
        m = 0
        for v in tr:
            m |= (1 << v)
        masks.append(m)
    best = None
    best_idx = None
    for idxs in combinations(range(len(triples)), 6):
        mx = max(idxs) + 1
        if best is not None and mx >= best:
            continue
        m = 0
        ok = True
        for i in idxs:
            mi = masks[i]
            if m & mi:
                ok = False
                break
            m |= mi
        if not ok or m != full:
            continue
        best = mx
        best_idx = idxs
    if best_idx is None:
        return [list(range(i, i+3)) for i in range(0, n, 3)]
    return [list(triples[i]) for i in best_idx]
