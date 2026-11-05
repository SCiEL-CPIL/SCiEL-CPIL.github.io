import time
import numpy as np

N = 17
NUM_BOOLEANS = 1 << N  # 131072

# Precompute per-wire bit masks as a boolean array of shape (NUM_BOOLEANS, N).
# _MASKS[k, i] is True iff wire i is 1 in the k-th Boolean input (input value = k).
_MASKS = ((np.arange(NUM_BOOLEANS, dtype=np.uint32)[:, None] >> np.arange(N, dtype=np.uint32)[None, :]) & 1).astype(bool)


def _check(network):
    """Return True iff `network` sorts every 2^N Boolean input (zero-one principle).

    Fast vectorized version: state[k, i] = value on wire i for input k.
    A compare-exchange on (i, j) sets state[:, i] = min, state[:, j] = max.
    """
    if not network:
        return N == 1
    state = _MASKS.copy()  # (NUM_BOOLEANS, N) bool
    for (i, j) in network:
        si = state[:, i]
        sj = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = si & sj
        state[:, j] = si | sj
    # Sorted ascending means no adjacent inversion: not (out[i]=1 and out[i+1]=0)
    for i in range(N - 1):
        if np.any(state[:, i] & ~state[:, i + 1]):
            return False
    return True


def _batcher_bitonic(n):
    """Return a correct bitonic sorting network for n wires (ascending)."""
    def comp(x, y):
        return (x, y) if x < y else (y, x)

    def bitonic_merge(a, d):
        if d == 1:
            yield comp(a[0], a[1])
            return
        for i in range(0, d, 2):
            yield comp(a[i], a[i + 1])
        yield from bitonic_merge(a[:d // 2], d // 2)
        yield from bitonic_merge(a[d // 2:], d // 2)

    def sort(a, d):
        if d == 1:
            return
        yield from sort(a[:d // 2], d // 2)
        yield from sort(a[d // 2:], d // 2)
        yield from bitonic_merge(a, d)

    return list(sort(list(range(n)), n))


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 82.0  # safety margin before the 90s kill

    # Start from a small, known-correct network: Batcher's bitonic (102 comparators for N=17).
    net = _batcher_bitonic(N)
    best_net = list(net)
    best_count = len(net)

    if not _check(net):
        # Fallback to the guaranteed-correct odd-even transposition seed (136 comparators).
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_net = list(net)
        best_count = len(net)
        if not _check(net):
            return best_net

    # Local search: repeatedly try to remove a comparator and keep the removal if
    # correctness is preserved. Use a random permutation of positions per pass to
    # avoid repeatedly probing the same slot; restart the permutation after each
    # successful removal so the indices stay valid.
    rng = np.random.default_rng(12345)
    while time.time() < deadline:
        order = rng.permutation(len(net))
        for pos in order:
            if time.time() >= deadline:
                break
            candidate = net[:pos] + net[pos + 1:]
            if _check(candidate):
                net = candidate
                best_net = list(net)
                best_count = len(net)
                # Indices shifted; break out to rebuild the permutation.
                break
    return best_net
# EVOLVE-BLOCK-END
