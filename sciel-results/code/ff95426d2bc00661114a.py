"""Search for a size-efficient 18-wire sorting network.

Entry point: construct() -> list of (i, j) integer pairs, 0 <= i < j < N.
"""
import time
import numpy as np

N = 18
NIN = 1 << N           # 262144 Boolean inputs
DEADLINE = 90.0        # wall-clock limit
MARGIN = 8.0           # safety margin before the limit


# ---------------- Batcher's odd-even merge sort (correct seed) ----------------
def _bit_length(n):
    return 1 + (n - 1).bit_length() if n > 1 else 1


def _odd_even_merge(n):
    """Return (comparators, depth) for an odd-even merge network on n wires."""
    if n == 1:
        return [], 0
    if n == 2:
        return [(0, 1)], 1
    if n % 2:
        comps, depth = _odd_even_merge(n - 1)
        return comps + [(i, n - 1) for i in range(n - 1)], depth + 1
    # even
    half = n // 2
    left, dl = _odd_even_merge(half)
    right, dr = _odd_even_merge(half)
    comps = list(left) + [(i + half, i + n - 1) for i in range(half)]
    depth = max(dl, dr) + 1
    # merge the two sorted halves
    comps += _merge_sorted(n)
    depth += _merge_depth(n)
    return comps, depth


def _merge_sorted(n):
    """Odd-even merge of two sorted halves of n//2 (n even)."""
    comps = []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    comps += [(i, i + half) for i in range(half)]
    # recursively merge the interleaved sub-networks
    comps += _merge_rec(n, 1)
    return comps


def _merge_rec(n, d):
    """Recursive odd-even merge network for n wires (n even), level d."""
    if n == 2:
        return []
    half = n // 2
    comps = []
    comps += [(i, i + half) for i in range(half)]
    comps += _merge_rec(half, d + 1)
    comps += _merge_rec(half, d + 1)
    return comps


def _merge_depth(n):
    if n == 2:
        return 1
    return 1 + _merge_depth(n // 2)


def _batcher_sort(n):
    """Return a correct Batcher odd-even merge sort network for n wires."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n % 2:
        comps, _ = _odd_even_merge(n)
        return comps
    half = n // 2
    left, _ = _odd_even_merge(half)
    right, _ = _odd_even_merge(half)
    comps = list(left) + [(i + half, i + n - 1) for i in range(half)]
    comps += _merge_sorted(n)
    return comps


# ---------------- Fast vectorized verification ----------------
def _verify(comps):
    """Return True iff `comps` sorts every Boolean input on N wires."""
    idx = np.arange(NIN, dtype=np.uint32)
    # wire k holds bit k of idx (0 = min, 1 = max).  min->0, max->1.
    for a, b in comps:
        ba = (idx >> a) & 1
        bb = (idx >> b) & 1
        # out_a = min = ba & bb ; out_b = max = ba | bb
        # write bit a to (ba & bb), bit b to (ba | bb)
        idx = (idx & ~(1 << a)) | ((ba & bb) << a)
        idx = (idx & ~(1 << b)) | ((ba | bb) << b)
    # sorted ascending: wire0=0 ... wire N-1=1  -> value 0b011..110 = NIN//2 - 1
    target = (NIN >> 1) - 1
    return bool(np.all(idx == target))


# ---------------- Search ----------------
def construct():
    start = time.time()
    deadline = start + (DEADLINE - MARGIN)

    base = _batcher_sort(N)
    if not _verify(base):
        # Fallback: odd-even transposition (always correct).
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    best = list(base)
    best_len = len(best)
    rng = np.random.default_rng(12345)

    def now_ok():
        return (time.time() - start) < deadline

    # Phase 1: greedy removal from the end (fastest win).
    changed = True
    while changed and now_ok():
        changed = False
        # scan from the end toward the front for the first removable comparator
        for pos in range(len(best) - 1, -1, -1):
            if not now_ok():
                break
            cand = best[:pos] + best[pos + 1:]
            if _verify(cand):
                best = cand
                best_len -= 1
                changed = True
                break  # restart scan (network changed)

    # Phase 2: non-adjacent position swaps (escape local minima).
    for _ in range(40000):
        if not now_ok():
            break
        n = len(best)
        if n < 2:
            break
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j or abs(i - j) == 1:
            continue
        cand = best[:]
        cand[i], cand[j] = best[j], best[i]
        if _verify(cand):
            best = cand

    # Phase 3: biased nearby-wire mutations (structural exploration).
    for _ in range(20000):
        if not now_ok():
            break
        n = len(best)
        pos = int(rng.integers(0, n))
        a, b = best[pos]
        # propose a new comparator, biased toward nearby wires
        lo = max(0, a - 4)
        hi = min(N - 1, b + 4)
        na = int(rng.integers(lo, hi + 1))
        nb = int(rng.integers(lo, hi + 1))
        if na == nb:
            continue
        if na > nb:
            na, nb = nb, na
        cand = best[:]
        cand[pos] = (na, nb)
        if _verify(cand):
            best = cand
            # after a mutation, try a few removals to harvest size
            for _k in range(8):
                if not now_ok():
                    break
                m = len(best)
                p = int(rng.integers(0, m))
                rc = best[:p] + best[p + 1:]
                if _verify(rc):
                    best = rc
                else:
                    break

    # Final: one more greedy sweep from the end to squeeze out leftovers.
    changed = True
    while changed and now_ok():
        changed = False
        for pos in range(len(best) - 1, -1, -1):
            if not now_ok():
                break
            cand = best[:pos] + best[pos + 1:]
            if _verify(cand):
                best = cand
                best_len = len(best)
                changed = True
                break

    return best


if __name__ == "__main__":
    net = construct()
    assert _verify(net)
    print("comparators:", len(net))
