import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START

# ---------- verification ----------
def _verify_network_fast(network, n=16):
    """Verify a network sorts all 2^n Boolean inputs (zero-one principle)."""
    if not network:
        return n <= 1
    for i, j in network:
        if i < 0 or i >= n or j < 0 or j >= n or i >= j:
            return False
    num_inputs = 1 << n
    inputs = np.arange(num_inputs, dtype=np.uint32)
    masks = (1 << np.arange(n, dtype=np.uint32))
    vals = ((inputs[:, None] & masks[None, :]) > 0).astype(np.uint8)
    for i, j in network:
        vi = vals[:, i]
        vj = vals[:, j]
        vals[:, i] = vi & vj
        vals[:, j] = vi | vj
    diffs = vals[:, 1:] - vals[:, :-1]
    return bool(np.all(diffs >= 0))


# ---------- Batcher Odd-Even Mergesort ----------
def _shift(comp, start):
    return (comp[0] + start, comp[1] + start)


def _sort(n, start, reverse=False):
    """Batcher Odd-Even Mergesort network on wires [start, start+n).
    Produces ascending order (or descending if reverse)."""
    if n <= 1:
        return []
    comps = []
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            length = min(2 * step, n - i)
            comps += _merge(start + i, length, reverse=reverse)
        step *= 2
    return comps


def _merge(start, n, reverse=False):
    """Bitonic merge on wires [start, start+n). Input must be bitonic:
    first n/2 ascending, last n/2 descending (in the given direction).
    Output: fully sorted in the given direction."""
    if n <= 1:
        return []
    if n == 2:
        # single comparator always puts min in lower index -> ascending
        if reverse:
            # can't reverse a single comparator; handled by caller
            return [(start, start + 1)]
        return [(start, start + 1)]
    half = n // 2
    comps = []
    # compare (i, i+half)
    for i in range(half):
        comps.append((start + i, start + i + half))
    # recurse on both halves
    comps += _merge(start, half, reverse=reverse)
    comps += _merge(start + half, half, reverse=reverse)
    return comps


def _descending_sort(n, start):
    """Network that sorts wires [start, start+n) in DESCENDING order."""
    # Build ascending sort on a reversed-index mapping:
    # local wire k maps to global (start + n - 1 - k).
    # An ascending sort network on local wires, when applied to the
    # reversed mapping, yields descending order on the original wires.
    asc = _sort(n, 0, reverse=False)
    desc = []
    for i, j in asc:
        gi = start + n - 1 - j
        gj = start + n - 1 - i
        if gi > gj:
            gi, gj = gj, gi
        desc.append((gi, gj))
    return desc


def construct():
    """Return a sorting network for 16 wires (Batcher Odd-Even Mergesort)."""
    t_start = time.time()
    deadline = t_start + 60.0  # safety margin vs 90s kill

    # Build the standard Batcher OES network for 16 wires.
    net = _sort(N, 0, reverse=False)

    # Verify; if somehow wrong, fall back to odd-even transposition.
    if not _verify_network_fast(net, N):
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best = net

    # Attempt bounded local improvement: try to drop redundant comparators.
    # Only proceed if we have ample time.
    if time.time() < deadline - 20.0 and len(net) > 0:
        best = _try_reduce(net, N, deadline)

    return best


def _try_reduce(net, n, deadline):
    """Greedily try to remove comparators that are not needed.
    Only removes a comparator if the network still sorts without it.
    Bounded by wall-clock deadline."""
    cur = list(net)
    i = 0
    while i < len(cur):
        if time.time() > deadline - 5.0:
            break
        trial = cur[:i] + cur[i + 1:]
        if _verify_network_fast(trial, n):
            cur = trial
            # do not increment i; check the same position again
        else:
            i += 1
    return cur

# EVOLVE-BLOCK-END
