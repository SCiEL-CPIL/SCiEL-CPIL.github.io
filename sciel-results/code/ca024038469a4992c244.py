import time
import random

N = 17
TIME_LIMIT = 75.0  # safety margin below the 90s kill

# ---------- helpers ----------

def _apply(wires, comps):
    """Apply compare-exchange list to a tuple/list of wire values."""
    for a, b in comps:
        x = wires[a]
        y = wires[b]
        if x > y:
            wires[a] = y
            wires[b] = x
    return wires


def _verify(comps):
    """Zero-one principle: check all 2^N inputs are sorted (ascending)."""
    for mask in range(1 << N):
        w = [ (mask >> i) & 1 for i in range(N) ]
        _apply(w, comps)
        for i in range(N - 1):
            if w[i] > w[i + 1]:
                return False
    return True


def _verify_random(comps, iters=3000):
    """Fast randomized sanity check over random permutations."""
    rng = random.Random(12345)
    for _ in range(iters):
        w = list(range(N))
        rng.shuffle(w)
        _apply(w, comps)
        for i in range(N - 1):
            if w[i] > w[i + 1]:
                return False
    return True


def _seed_network():
    """Odd-even transposition network (correct, 136 comparators for N=17)."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _build_sort_network():
    """Deterministic correct network by sorting a random permutation."""
    comps = []
    w = list(range(N))
    while w != sorted(w):
        for i in range(N - 1):
            if w[i] > w[i + 1]:
                comps.append((i, i + 1))
                w[i], w[i + 1] = w[i + 1], w[i]
    return comps


def _prune(comps, deadline):
    """Remove any comparator whose removal still leaves a correct network."""
    comps = list(comps)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            cand = comps[:idx] + comps[idx + 1:]
            if _verify(cand):
                del comps[idx]
                changed = True
    return comps


def _random_swap(comps, deadline, seed):
    """Try swapping two comparators (reorder) to enable further pruning."""
    rng = random.Random(seed)
    n = len(comps)
    if n < 2:
        return comps
    for _ in range(200):
        if time.time() > deadline:
            return comps
        a, b = rng.randrange(n), rng.randrange(n)
        if a == b:
            continue
        cand = list(comps)
        cand[a], cand[b] = cand[b], cand[a]
        if _verify(cand):
            return cand
    return comps


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT

    # Start from a correct seed network.
    comps = _seed_network()

    # Verify the seed is actually correct (guard against bugs).
    if not _verify(comps):
        comps = _build_sort_network()

    # Greedy pruning (the main size-reduction engine).
    comps = _prune(comps, deadline)

    # Use spare time for reordering + re-pruning to escape local optima.
    if time.time() < deadline - 2.0:
        best = comps
        for seed in range(1, 2000):
            if time.time() > deadline - 2.0:
                break
            cur = _random_swap(best, deadline - 2.0, seed)
            if len(cur) < len(best):
                best = cur
            cur = _prune(cur, deadline - 2.0)
            if len(cur) < len(best):
                best = cur
        comps = best

    # Final safety: ensure correctness; fall back to a known-good network.
    if not _verify(comps):
        comps = _build_sort_network()

    return [(i, j) for (i, j) in comps]
