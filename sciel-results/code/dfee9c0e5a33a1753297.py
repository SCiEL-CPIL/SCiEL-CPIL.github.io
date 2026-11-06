"""Additive bases with few representations - optimized construction."""
import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
NUM_TARGETS = 129  # targets 128..256 inclusive

_START_TIME = time.time()
DEADLINE = 85.0  # safety margin before 90s kill


def _elapsed():
    return time.time() - _START_TIME


def _pair_counts(A):
    """Compute representation counts for each target in [128,256].
    A is a sorted list of ints in [0,128].
    Returns a list of 129 counts (index 0 -> target 128, index 128 -> target 256).
    """
    counts = [0] * NUM_TARGETS
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            if TARGET_LO <= s <= TARGET_HI:
                counts[s - TARGET_LO] += 1
    return counts


def _peak(counts):
    return max(counts)


def _is_valid(counts):
    return all(c > 0 for c in counts)


def _try_swap(A, A_set, counts, i, x_new):
    """Try replacing A[i] with x_new. Returns (new_counts, new_peak) or None if invalid."""
    ai = A[i]
    if x_new == ai:
        return None
    if x_new in A_set:
        return None

    n = len(A)
    new_counts = list(counts)

    # Remove contributions of ai
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1

    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1

    # Add contributions of x_new
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] += 1

    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = x_new + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] += 1

    if any(c < 0 for c in new_counts):
        return None

    if any(c == 0 for c in new_counts):
        return None

    peak = max(new_counts)
    return (new_counts, peak)


def _local_search(A, max_time=80.0):
    """Perform swap-based local search to minimize peak multiplicity.
    Returns (best_A, best_peak, best_counts).
    """
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return (A, float('inf'), counts)

    current_peak = _peak(counts)
    best_A = list(A)
    best_peak = current_peak
    best_counts = list(counts)

    n = len(A)
    improved = True
    stall_count = 0
    max_stall = 50

    while improved and _elapsed() < max_time:
        improved = False
        stall_count += 1

        for i in range(n):
            if _elapsed() > max_time:
                break
            ai = A[i]
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_peak = result
                if new_peak < current_peak:
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    stall_count = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                        best_counts = list(counts)
                    break

        if not improved and stall_count >= max_stall:
            break

    return (best_A, best_peak, best_counts)


def _try_remove(A, A_set, counts, i):
    """Try removing A[i]. Returns (new_counts) or None if invalid."""
    ai = A[i]
    n = len(A)
    new_counts = list(counts)

    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1

    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1

    if any(c <= 0 for c in new_counts):
        return None

    return new_counts


def _reduce_size(A, max_time=80.0):
    """Try to remove redundant elements while maintaining coverage."""
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return A, counts

    changed = True
    while changed and _elapsed() < max_time and len(A) > 15:
        changed = False
        for i in range(len(A)):
            if _elapsed() > max_time:
                break
            result = _try_remove(A, A_set, counts, i)
            if result is not None:
                ai = A[i]
                A.pop(i)
                A_set.discard(ai)
                counts = result
                changed = True
                break

    return A, counts


def _generate_initial_A(size, rng):
    """Generate an initial random set of given size from [0, 128]."""
    return sorted(rng.sample(range(0, N + 1), size))


def _fix_coverage(A, counts, rng, max_adds=20):
    """Add elements to cover any uncovered targets."""
    A = sorted(A)
    for _ in range(max_adds):
        if _is_valid(counts):
            break
        uncovered = [i for i in range(NUM_TARGETS) if counts[i] == 0]
        if not uncovered:
            break

        min_target_idx = min(uncovered, key=lambda i: counts[i])
        min_target = TARGET_LO + min_target_idx

        best_x = None
        best_gain = -1
        A_set = set(A)
        for x in range(0, N + 1):
            if x in A_set:
                continue
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                    gain += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None or best_gain <= 0:
            candidates = [x for x in range(0, N + 1) if x not in A_set]
            if not candidates:
                break
            best_x = rng.choice(candidates)

        A.append(best_x)
        A.sort()
        counts = _pair_counts(A)

    return A, counts


def _balance_polish(A, max_time=80.0):
    """
    High-precision polish: after local search converges to a local optimum on peak,
    try to reduce the variance of counts (make distribution more uniform) without
    increasing the peak. This can sometimes enable a subsequent peak reduction.
    Uses a surrogate objective: minimize sum of (count - mean)^2 subject to peak not increasing.
    """
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return (A, float('inf'), counts)

    current_peak = _peak(counts)
    best_A = list(A)
    best_peak = current_peak
    best_counts = list(counts)
    best_variance = float(np.var(counts))

    n = len(A)

    # Iterate: try swaps that reduce variance while keeping peak <= current_peak
    # and also try swaps that reduce peak
    max_iterations = 500
    for _ in range(max_iterations):
        if _elapsed() > max_time:
            break

        found_improvement = False
        for i in range(n):
            if _elapsed() > max_time:
                break
            ai = A[i]
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_peak = result

                # Accept if peak decreases, or peak same but variance decreases
                if new_peak < current_peak:
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_peak = new_peak
                    found_improvement = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                        best_counts = list(counts)
                    break
                elif new_peak == current_peak:
                    new_var = float(np.var(new_counts))
                    if new_var < best_variance - 1e-6:
                        A[i] = x_new
                        A_set.discard(ai)
                        A_set.add(x_new)
                        A.sort()
                        counts = new_counts
                        best_variance = new_var
                        found_improvement = True
                        best_A = list(A)
                        best_counts = list(counts)
                        break

            if found_improvement:
                break

        if not found_improvement:
            break

    return (best_A, best_peak, best_counts)


def _simulated_annealing_polish(A, counts, rng, max_time=80.0):
    """
    Simulated annealing on the peak objective with a smooth surrogate.
    Accept moves that increase peak with probability exp(-delta/T) to escape local optima.
    """
    A = sorted(A)
    A_set = set(A)
    counts = list(counts)

    if not _is_valid(counts):
        return (A, float('inf'), counts)

    current_peak = _peak(counts)
    best_A = list(A)
    best_peak = current_peak
    best_counts = list(counts)

    n = len(A)
    T = 2.0  # initial temperature
    cooling = 0.995
    min_T = 0.01

    while _elapsed() < max_time and T > min_T:
        # Pick random swap
        i = rng.randint(0, n - 1)
        ai = A[i]
        candidates = [x for x in range(0, N + 1) if x != ai and x not in A_set]
        if not candidates:
            break
        x_new = rng.choice(candidates)

        result = _try_swap(A, A_set, counts, i, x_new)
        if result is None:
            T *= cooling
            continue

        new_counts, new_peak = result
        delta = new_peak - current_peak

        if delta <= 0:
            # Accept
            A[i] = x_new
            A_set.discard(ai)
            A_set.add(x_new)
            A.sort()
            counts = new_counts
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = list(A)
                best_counts = list(counts)
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta / T):
                A[i] = x_new
                A_set.discard(ai)
                A_set.add(x_new)
                A.sort()
                counts = new_counts
                current_peak = new_peak

        T *= cooling

    return (best_A, best_peak, best_counts)


def _greedy_balanced_build(rng, max_time=80.0):
    """
    Build a set greedily with balance awareness: at each step, choose the element
    that minimizes the resulting peak, with tie-breaking by variance reduction.
    This is the load-balancing / minimum-maximum flow heuristic adapted to this problem.
    """
    A = []
    A_set = set()
    counts = [0] * NUM_TARGETS

    # Start with a structured seed: spread elements across [0, 128]
    seed_size = 15
    # Use a quasi-random spread
    for k in range(seed_size):
        x = (k * 129) // seed_size  # spread across [0, 128]
        if x not in A_set:
            A.append(x)
            A_set.add(x)
    A.sort()
    counts = _pair_counts(A)

    target_size = 40

    while len(A) < target_size and _elapsed() < max_time:
        best_x = None
        best_peak = float('inf')
        best_var = float('inf')
        best_new_counts = None

        for x in range(0, N + 1):
            if x in A_set:
                continue

            # Compute new counts
            new_counts = list(counts)
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI:
                    new_counts[s - TARGET_LO] += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI:
                new_counts[s - TARGET_LO] += 1

            # Must maintain coverage
            if any(c == 0 for c in new_counts):
                continue

            pk = max(new_counts)
            if pk < best_peak or (pk == best_peak and float(np.var(new_counts)) < best_var):
                best_peak = pk
                best_var = float(np.var(new_counts))
                best_x = x
                best_new_counts = new_counts

        if best_x is None:
            # Relax: allow adding even if it creates a zero (we'll fix later)
            best_x = None
            best_peak = float('inf')
            for x in range(0, N + 1):
                if x in A_set:
                    continue
                new_counts = list(counts)
                for y in A:
                    s = x + y
                    if TARGET_LO <= s <= TARGET_HI:
                        new_counts[s - TARGET_LO] += 1
                s = x + x
                if TARGET_LO <= s <= TARGET_HI:
                    new_counts[s - TARGET_LO] += 1
                pk = max(new_counts)
                if pk < best_peak:
                    best_peak = pk
                    best_x = x
                    best_new_counts = new_counts

            if best_x is None:
                break

        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = best_new_counts

    return A, counts


def construct():
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_A = None
    best_peak = float('inf')
    best_counts = None

    # Phase 1: Greedy balanced build + local search + polish
    if _elapsed() < DEADLINE - 30:
        A_g, counts_g = _greedy_balanced_build(rng, max_time=DEADLINE - 40)
        if _is_valid(counts_g):
            A_g, pk_g, cnt_g = _local_search(A_g, max_time=DEADLINE - 30)
            if _is_valid(cnt_g):
                A_g, pk_g, cnt_g = _balance_polish(A_g, max_time=DEADLINE - 20)
                if _is_valid(cnt_g):
                    if pk_g < best_peak:
                        best_peak = pk_g
                        best_A = list(A_g)
                        best_counts = list(cnt_g)

    # Phase 2: Multiple restarts with random initial sets
    sizes_to_try = [32, 34, 36, 38, 40, 42, 44, 46, 48]
    restarts_per_size = 2

    for size in sizes_to_try:
        if _elapsed() > DEADLINE - 15:
            break

        for restart in range(restarts_per_size):
            if _elapsed() > DEADLINE - 10:
                break

            A = _generate_initial_A(size, rng)
            counts = _pair_counts(A)

            if not _is_valid(counts):
                A, counts = _fix_coverage(A, counts, rng)

            if not _is_valid(counts):
                continue

            peak = _peak(counts)

            # Local search
            A_s, pk_s, cnt_s = _local_search(A, max_time=DEADLINE - 10)
            if not _is_valid(cnt_s):
                continue
            pk_s = _peak(cnt_s)

            # Balance polish
            A_s, pk_s, cnt_s = _balance_polish(A_s, max_time=DEADLINE - 8)
            if _is_valid(cnt_s):
                pk_s = _peak(cnt_s)

            # Simulated annealing polish (short)
            if _elapsed() < DEADLINE - 5 and pk_s <= best_peak:
                A_s, pk_s, cnt_s = _simulated_annealing_polish(A_s, cnt_s, rng, max_time=DEADLINE - 5)
                if _is_valid(cnt_s):
                    pk_s = _peak(cnt_s)

            # Reduce size
            A_s, cnt_s = _reduce_size(A_s, max_time=DEADLINE - 3)
            if _is_valid(cnt_s):
                pk_s = _peak(cnt_s)

            if pk_s < best_peak:
                best_peak = pk_s
                best_A = list(A_s)
                best_counts = list(cnt_s)

            if best_peak <= 3:
                break

        if best_peak <= 3:
            break

    # Final: if we have a valid result, do one more polish pass
    if best_A is not None and _is_valid(best_counts) and _elapsed() < DEADLINE - 3:
        A_f, pk_f, cnt_f = _balance_polish(best_A, max_time=DEADLINE - 3)
        if _is_valid(cnt_f) and pk_f < best_peak:
            best_peak = pk_f
            best_A = list(A_f)
            best_counts = list(cnt_f)

    # Fallback
    if best_A is None or not _is_valid(best_counts):
        best_A = list(range(0, 129, 2))
        best_counts = _pair_counts(best_A)
        if not _is_valid(best_counts):
            best_A = list(range(0, 129))
            best_counts = _pair_counts(best_A)
        best_peak = _peak(best_counts)

    return best_A
