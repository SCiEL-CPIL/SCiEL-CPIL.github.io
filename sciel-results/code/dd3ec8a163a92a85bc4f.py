"""Sorting-network optimizer for 18 wires.

Starts from the odd-even transposition network (153 comparators) and
reduces its size using vectorized NumPy verification over all 2^18 Boolean
inputs, combined with greedy comparator removal and 1-move local search.
A hard wall-clock deadline guard returns the best valid network found.
"""
import time
import numpy as np

N = 18
TIME_LIMIT = 60.0          # evaluator hard limit
SAFETY = 10.0              # keep a margin -> effective deadline
DEADLINE = 0.0             # set at construct() start


def _apply_state(state, comp):
    """Apply one compare-exchange (i,j) to a boolean state array (vectorized)."""
    i, j = comp
    a = state[:, i]
    b = state[:, j]
    # min -> wire i, max -> wire j
    state[:, i] = np.minimum(a, b)
    state[:, j] = np.maximum(a, b)
    return state


def _verify_all(comps):
    """Return True iff the network sorts every one of the 2^N Boolean inputs."""
    # All 2^N bit patterns as columns; rows = inputs, cols = wires.
    bits = np.arange(1 << N, dtype=np.uint8)
    state = np.zeros((1 << N, N), dtype=np.uint8)
    for w in range(N):
        state[:, w] = (bits >> w) & 1
    for comp in comps:
        _apply_state(state, comp)
    # sorted ascending: wire 0 <= wire 1 <= ... <= wire N-1  (booleans)
    # i.e. no position where a 1 is followed by a 0.
    return not np.any(state[:, :-1] > state[:, 1:])


def _verify_subset(comps, subset):
    """Verify only a subset of inputs (for cheap checks)."""
    bits = np.array(subset, dtype=np.uint8)
    state = np.zeros((len(bits), N), dtype=np.uint8)
    for w in range(N):
        state[:, w] = (bits >> w) & 1
    for comp in comps:
        _apply_state(state, comp)
    return not np.any(state[:, :-1] > state[:, 1:])


def _full_network():
    """Odd-even transposition network on N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def construct():
    global DEADLINE
    start = time.time()
    DEADLINE = start + SAFETY

    def time_ok():
        return (time.time() - start) < (SAFETY - 2.0)

    # Baseline: full odd-even transposition network.
    comps = _full_network()
    if not _verify_all(comps):
        # Should never happen; return as-is.
        return comps

    best = list(comps)
    best_size = len(best)

    # Precompute a representative subset of inputs for cheap checks.
    # Include all "sorted-ish" patterns, random samples, and extremes.
    rng = np.random.default_rng(12345)
    subset_list = []
    # Extremes and structured patterns
    subset_list.append(0)
    subset_list.append((1 << N) - 1)
    for p in range(N):
        subset_list.append(1 << p)
        subset_list.append((1 << N) - 1 - (1 << p))
    # Random samples
    subset_list.extend(rng.integers(0, 1 << N, size=4000, dtype=np.uint8).tolist())
    # De-duplicate
    subset = sorted(set(subset_list))

    # ---------------- Phase 1: Greedy removal ----------------
    # Try to remove each comparator (in reverse order, then forward),
    # keeping the network if it still sorts the subset, then confirm with full check.
    for sweep in range(8):
        if not time_ok():
            break
        # alternate direction for diversity
        indices = range(len(comps) - 1, -1, -1) if sweep % 2 == 0 else range(len(comps))
        i = 0
        for idx in indices:
            if not time_ok():
                break
            if idx >= len(comps):
                continue
            trial = comps[:idx] + comps[idx + 1:]
            if _verify_subset(trial, subset):
                if _verify_all(trial):
                    comps = trial
                    # continue scanning from the same position (size shrank)
                # else keep it
        if len(comps) < best_size:
            best_size = len(comps)
            best = list(comps)

    # ---------------- Phase 2: 1-move local search ----------------
    # Try moving a single comparator to a different position, or replacing it
    # with a different valid pair, to enable further removals.
    def try_replace_and_shrink(comps):
        """Try replacing each comparator with a candidate pair; if a swap
        keeps validity and later allows a removal, keep it."""
        nonlocal best_size, best
        # Candidate pairs: all (i,j), i<j
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        for pos in range(len(comps)):
            if not time_ok():
                return comps
            cur = comps[pos]
            # skip trying the same pair
            for cand in all_pairs:
                if cand == cur:
                    continue
                trial = comps[:pos] + [cand] + comps[pos + 1:]
                if _verify_subset(trial, subset):
                    if _verify_all(trial):
                        # attempt one removal pass on this improved network
                        shrunken = _quick_shrink(trial)
                        if len(shrunken) < len(comps):
                            return shrunken
                        return trial
        return comps

    def _quick_shrink(comps):
        """One pass of greedy removal on comps."""
        for idx in range(len(comps) - 1, -1, -1):
            if not time_ok():
                return comps
            trial = comps[:idx] + comps[idx + 1:]
            if _verify_subset(trial, subset) and _verify_all(trial):
                comps = trial
        return comps

    for ls_iter in range(40):
        if not time_ok():
            break
        new_comps = try_replace_and_shrink(comps)
        if len(new_comps) < len(comps):
            comps = new_comps
            if len(comps) < best_size:
                best_size = len(comps)
                best = list(comps)
        else:
            # no improvement; try a small random perturbation to escape
            if len(comps) > 2:
                pos = rng.integers(0, len(comps))
                all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                cand = all_pairs[rng.integers(0, len(all_pairs))]
                trial = comps[:pos] + [cand] + comps[pos + 1:]
                if _verify_subset(trial, subset) and _verify_all(trial):
                    shrunken = _quick_shrink(trial)
                    comps = shrunken
                    if len(comps) < best_size:
                        best_size = len(comps)
                        best = list(comps)

    # ---------------- Final: return best valid network ----------------
    # Ensure best is valid (it was verified when set). Return it.
    if len(best) <= len(comps):
        return best
    return comps


# EVOLVE-BLOCK-END
