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
    # Pair (ai, ai) - counted once in the (i,i) slot
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1

    # Pairs (ai, aj) for j != i
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

    # Check validity - no negative counts
    if any(c < 0 for c in new_counts):
        return None

    # Check coverage
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

        # Try all possible swaps
        for i in range(n):
            if _elapsed() > max_time:
                break
            ai = A[i]
            # Try replacing with each value in [0, 128] not in A
            # Focus on values that might help reduce the peak
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_peak = result
                if new_peak < current_peak:
                    # Accept
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
                    break  # restart from beginning after a move

        if not improved and stall_count >= max_stall:
            break

    return (best_A, best_peak, best_counts)


def _try_remove(A, A_set, counts, i):
    """Try removing A[i]. Returns (new_counts) or None if invalid."""
    ai = A[i]
    n = len(A)
    new_counts = list(counts)

    # Remove pair (ai, ai)
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1

    # Remove pairs (ai, aj) for j != i
    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1

    # Check coverage
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

        # Find the target with minimum count
        min_target_idx = min(uncovered, key=lambda i: counts[i])
        min_target = TARGET_LO + min_target_idx

        # Find a candidate x that helps cover min_target
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
            # Pick a random element
            candidates = [x for x in range(0, N + 1) if x not in A_set]
            if not candidates:
                break
            best_x = rng.choice(candidates)

        A.append(best_x)
        A.sort()
        counts = _pair_counts(A)

    return A, counts


def _greedy_build(rng, max_time=80.0):
    """Build a set greedily by adding elements that minimize peak."""
    A = []
    A_set = set()
    counts = [0] * NUM_TARGETS

    # Start with a few random elements
    initial_size = 10
    for _ in range(initial_size):
        x = rng.randint(0, N)
        if x not in A_set:
            A.append(x)
            A_set.add(x)

    A.sort()
    counts = _pair_counts(A)

    # Greedily add elements
    while _elapsed() < max_time:
        # Find the current peak
        current_peak = _peak(counts)

        # Try each candidate x not in A
        best_x = None
        best_new_peak = current_peak
        best_new_counts = None

        for x in range(0, N + 1):
            if x in A_set:
                continue

            # Compute new counts if we add x
            new_counts = list(counts)
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI:
                    new_counts[s - TARGET_LO] += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI:
                new_counts[s - TARGET_LO] += 1

            if not _is_valid(new_counts):
                continue

            new_peak = _peak(new_counts)
            if new_peak < best_new_peak:
                best_new_peak = new_peak
                best_x = x
                best_new_counts = new_counts

        if best_x is None or best_new_peak >= current_peak:
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

    # Strategy: try various sizes, use local search
    # Smaller sets have fewer pairs, potentially more uniform distribution
    sizes_to_try = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]

    # For each size, do multiple restarts
    restarts_per_size = 3

    for size in sizes_to_try:
        if _elapsed() > DEADLINE - 10:
            break

        for restart in range(restarts_per_size):
            if _elapsed() > DEADLINE - 5:
                break

            # Generate initial set
            A = _generate_initial_A(size, rng)
            counts = _pair_counts(A)

            # If not valid, try to fix by adding elements
            if not _is_valid(counts):
                A, counts = _fix_coverage(A, counts, rng)

            if not _is_valid(counts):
                continue

            peak = _peak(counts)

            # Local search
            A_sorted, peak, counts = _local_search(A, max_time=DEADLINE - 5)

            # Try to reduce size
            A_sorted, counts = _reduce_size(A_sorted, max_time=DEADLINE - 5)
            if _is_valid(counts):
                peak = _peak(counts)

            if peak < best_peak:
                best_peak = peak
                best_A = list(A_sorted)
                best_counts = list(counts)

            # Early termination if we hit a good peak
            if best_peak <= 3:
                break

        if best_peak <= 3:
            break

    # Also try greedy build
    if _elapsed() < DEADLINE - 5:
        A_greedy, counts_greedy = _greedy_build(rng, max_time=DEADLINE - 10)
        if _is_valid(counts_greedy):
            peak_greedy = _peak(counts_greedy)
            A_greedy, peak_greedy, counts_greedy = _local_search(A_greedy, max_time=DEADLINE - 5)
            A_greedy, counts_greedy = _reduce_size(A_greedy, max_time=DEADLINE - 5)
            if _is_valid(counts_greedy):
                peak_greedy = _peak(counts_greedy)
                if peak_greedy < best_peak:
                    best_peak = peak_greedy
                    best_A = list(A_greedy)
                    best_counts = list(counts_greedy)

    # If we still don't have a valid set, fall back to a simple construction
    if best_A is None or not _is_valid(best_counts):
        # Use a structured set
        best_A = list(range(0, 129, 2))  # all even numbers
        best_counts = _pair_counts(best_A)
        if not _is_valid(best_counts):
            # Fallback: all numbers
            best_A = list(range(0, 129))
            best_counts = _pair_counts(best_A)
        best_peak = _peak(best_counts)

    return best_A
