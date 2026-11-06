import random
import time
import numpy as np

N = 17
MASK = (1 << N) - 1
# Precompute the 2^N zero-one masks (131072 ints).
_ALL = np.arange(1 << N, dtype=np.int64)
# Precompute, for every mask and every possible comparator (i,j), the result mask.
# comp[i,j] = array of length 2^N: result of applying (i,j) to each mask.
_COMP = {}
for i in range(N):
    for j in range(i + 1, N):
        li, lj = 1 << i, 1 << j
        a = np.bitwise_and(_ALL, li)
        b = np.bitwise_and(_ALL, lj)
        # if bit i set and bit j clear -> swap
        swap = (a & ~b)
        res = _ALL.copy()
        # clear i and set j where swap
        res = np.where(swap, res ^ (li | lj), res)
        _COMP[(i, j)] = res

def _apply(net, masks):
    for (i, j) in net:
        masks = _COMP[(i, j)][masks]
    return masks

def _is_sorted_mask(m):
    # sorted ascending means 1s are on the low wires (min to wire i)
    # i.e. the set of 1-bits is a prefix of low bits
    return (m & (m + 1)) == 0

def _verify(net):
    # apply to all masks, check sorted
    m = _ALL.copy()
    for (i, j) in net:
        m = _COMP[(i, j)][m]
    return bool(np.all(_is_sorted_mask(m)))

def _verify_random(net, k=20000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(k):
        m = rng.integers(0, 1 << N)
        for (i, j) in net:
            li, lj = 1 << i, 1 << j
            if (m & li) and not (m & lj):
                m ^= (li | lj)
        if (m & (m + 1)) != 0:
            return False
    return True

def _depth(net):
    d = [0] * N
    for (i, j) in net:
        nd = d[i] + 1
        if nd > d[j]:
            d[j] = nd
    return max(d)

def _merge_sort_net(n):
    """Build a correct sorting network for n wires (merge-insertion)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    # split
    m = n // 2
    left = _merge_sort_net(m)
    right = _merge_sort_net(n - m)
    # right shifted by m
    right = [(a + m, b + m) for (a, b) in right]
    net = left + right
    # merge: for k in 0..n-m-1, insert wire (m+k) into sorted prefix [0..m+k-1]
    for k in range(n - m):
        t = m + k
        # insert t into [0..t-1]
        for p in range(t - 1, -1, -1):
            net.append((p, t))
            net.append((p, p + 1))
    return net

def _build():
    net = _merge_sort_net(N)
    # dedupe consecutive duplicates
    seen = set()
    ded = []
    for c in net:
        if c not in seen:
            seen.add(c)
            ded.append(c)
    return ded

def _try_remove(net):
    n0 = len(net)
    for idx in range(n0 - 1, -1, -1):
        cand = net[:idx] + net[idx + 1:]
        if _verify(cand):
            return cand
    return None

def _try_swap(net):
    n0 = len(net)
    for a in range(n0):
        for b in range(a + 1, n0):
            cand = net[:]
            cand[a], cand[b] = net[b], net[a]
            if _verify(cand):
                return cand
    return None

def construct():
    deadline = time.time() + 80.0
    net = _build()
    assert _verify(net)
    best = net
    rng = random.Random(12345)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # removals
        r = _try_remove(best)
        if r is not None:
            best = r
            improved = True
            continue
        # swaps (limited)
        s = _try_swap(best)
        if s is not None:
            best = s
            improved = True
            continue
        # random local search: random reorder + removals
        for _ in range(200):
            if time.time() >= deadline:
                break
            cand = best[:]
            rng.shuffle(cand)
            # remove duplicates
            seen = set()
            ded = []
            for c in cand:
                if c not in seen:
                    seen.add(c)
                    ded.append(c)
            cand = ded
            # try removals
            changed = True
            while changed:
                changed = False
                n0 = len(cand)
                for idx in range(n0 - 1, -1, -1):
                    c2 = cand[:idx] + cand[idx + 1:]
                    if _verify(c2):
                        cand = c2
                        changed = True
                        break
            if len(cand) < len(best):
                best = cand
                improved = True
                break
    return best
