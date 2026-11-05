"""Size-efficient sorting networks on 17 wires via local search over Boolean (zero-one) verification."""
import time
import numpy as np

N = 17
NUM_BOOL = 1 << N  # 131072


def _now():
    return time.time()


def _seed_network():
    # Odd-even transposition sorting network: N phases, 2*(N-1) comparators.
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _simulate(net, bools):
    """Apply network to a (NUM_BOOL, N) uint8 array in place; returns it."""
    for (a, b) in net:
        va = bools[:, a]
        vb = bools[:, b]
        bools[:, a] = np.minimum(va, vb)
        bools[:, b] = np.maximum(va, vb)
    return bools


def _build_bools():
    """All 2^N Boolean inputs as a (NUM_BOOL, N) uint8 matrix."""
    idx = np.arange(NUM_BOOL, dtype=np.uint32)
    bools = np.zeros((NUM_BOOL, N), dtype=np.uint8)
    for bit in range(N):
        bools[:, bit] = ((idx >> bit) & 1).astype(np.uint8)
    return bools


def _is_sorted(arr):
    if N <= 1:
        return True
    return not np.any(arr[:, :-1] > arr[:, 1:])


def _verify(net, bools):
    work = bools.copy()
    _simulate(net, work)
    return _is_sorted(work)


def construct():
    deadline = _now() + 85.0
    best = _seed_network()
    bools = _build_bools()

    if not _verify(best, bools):
        fallback = []
        for phase in range(2 * N):
            for i in range(phase % 2, N - 1, 2):
                fallback.append((i, i + 1))
        best = fallback

    # Local search: alternate deletion and swap passes until no improvement.
    while _now() < deadline:
        progress = False

        # Deletion pass: remove one comparator if the network still sorts.
        for k in range(len(best)):
            if _now() >= deadline:
                break
            cand = best[:k] + best[k + 1:]
            if _verify(cand, bools):
                best = cand
                progress = True
                break  # restart deletion scan with shorter network

        if _now() >= deadline:
            break

        # Swap pass: try a few adjacent swaps to escape local minima.
        # Limit to a small number of swap attempts per round to avoid stalling.
        for _ in range(3):
            if _now() >= deadline:
                break
            swapped = False
            for k in range(len(best) - 1):
                if _now() >= deadline:
                    break
                cand = best[:k] + [best[k + 1], best[k]] + best[k + 2:]
                if _verify(cand, bools):
                    best = cand
                    swapped = True
                    break
            if not swapped:
                break

        if not progress:
            break

    return best
