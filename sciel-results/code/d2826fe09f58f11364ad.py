"""Distinct subset sums, n=12, bounded max.
Strategy: randomized local search starting from a random valid superincreasing-like
sequence, with mutations that swap or adjust elements to reduce the maximum value
while verifying distinct subset sums via a fast bit-DP check.
"""
import time
import random
import sys

N = 12
NSUB = 1 << N
MAXV = 1048576
DEADLINE = 82.0          # hard stop well under the 90 s kill
DEADLINE_SOFT = 75.0     # stop searching once a good solution is in hand

# ---- fast subset-sum distinctness check ----
def _all_sums_distinct(arr):
    """Return True if all 2^N subset sums of arr are distinct."""
    n = len(arr)
    sums = [0] * (1 << n)
    for i in range(n):
        b = arr[i]
        h = 1 << i
        for j in range(h):
            sums[h | j] = sums[j] + b
    # check distinctness
    if len(set(sums)) != NSUB:
        return False
    return True


def _max_val(arr):
    return max(arr)


def _valid(arr):
    """Check: exactly N distinct positive ints <= MAXV, all subset sums distinct."""
    if len(arr) != N:
        return False
    if any(v <= 0 for v in arr):
        return False
    if len(set(arr)) != N:
        return False
    if max(arr) > MAXV:
        return False
    return _all_sums_distinct(arr)


def _construct_random_superincreasing(rng, max_bound):
    """Generate a random superincreasing sequence (guaranteed distinct subset sums)."""
    arr = []
    cur = 1
    for i in range(N):
        # pick a value at least cur (superincreasing property)
        low = cur
        high = min(max_bound, max(low, cur * 2))
        if low > high:
            low = high = cur
        v = rng.randint(low, high)
        arr.append(v)
        cur = sum(arr) + 1  # next must exceed sum of all previous
    return arr


def _local_search(rng, start, deadline):
    """
    Local search: start from a valid set, try mutations to reduce max element.
    Mutations:
      1. Decrease the largest element by a small random amount.
      2. Swap two elements.
      3. Replace one element with a smaller random value.
      4. Randomly perturb a non-max element.
    """
    best = list(start)
    best_max = _max_val(best)
    t0 = time.time()
    stuck = 0

    while time.time() < deadline:
        arr = list(best)
        # Choose a mutation strategy
        strat = rng.randint(0, 3)

        if strat == 0:
            # Decrease the largest element
            idx = arr.index(best_max)
            delta = rng.randint(1, max(1, best_max // 10))
            new_val = arr[idx] - delta
            if new_val < 1:
                new_val = 1
            if new_val == arr[idx]:
                continue
            arr[idx] = new_val
        elif strat == 1:
            # Swap two elements
            i, j = rng.sample(range(N), 2)
            arr[i], arr[j] = arr[j], arr[i]
        elif strat == 2:
            # Replace a random element with a smaller value
            idx = rng.randint(0, N - 1)
            new_val = rng.randint(1, arr[idx])
            arr[idx] = new_val
        else:
            # Randomly perturb: pick two elements, adjust one down, maybe another up
            i, j = rng.sample(range(N), 2)
            if arr[i] > 1:
                arr[i] -= rng.randint(1, max(1, arr[i] // 5))
                if arr[i] < 1:
                    arr[i] = 1
            arr[j] += rng.randint(1, max(1, best_max // 20))

        # Check validity
        if len(set(arr)) != N or any(v <= 0 for v in arr):
            continue
        if max(arr) > MAXV:
            continue
        if not _all_sums_distinct(arr):
            stuck += 1
            continue

        new_max = _max_val(arr)
        if new_max < best_max:
            best = arr
            best_max = new_max
            stuck = 0
        else:
            stuck += 1

        # If stuck too long, do a random restart from a new superincreasing base
        if stuck > 500:
            # Try a fresh random superincreasing sequence with tighter bound
            base = _construct_random_superincreasing(rng, best_max)
            if _valid(base) and _max_val(base) < best_max:
                best = base
                best_max = _max_val(best)
            stuck = 0

    return best


def _greedy_min_max(rng, deadline):
    """
    Greedy construction: build the set element by element, choosing the smallest
    possible value at each step that keeps all subset sums distinct.
    """
    arr = []
    for i in range(N):
        # Try values starting from 1 upward
        found = False
        # Upper bound: don't go too high
        upper = MAXV
        for v in range(1, upper + 1):
            if time.time() >= deadline:
                return None
            candidate = arr + [v]
            if len(set(candidate)) != len(candidate):
                continue
            if _all_sums_distinct(candidate):
                arr.append(v)
                found = True
                break
        if not found:
            return None
    return arr


def construct():
    t0 = time.time()
    rng = random.Random(42)

    # Known-good fallback: powers of two (max 2048), always valid.
    best = [1 << i for i in range(N)]
    best_max = max(best)

    # Phase 1: Greedy construction to get a good starting point
    deadline_phase1 = t0 + 10.0
    greedy_result = _greedy_min_max(rng, deadline_phase1)
    if greedy_result is not None and _valid(greedy_result):
        if max(greedy_result) < best_max:
            best = greedy_result
            best_max = max(best)

    # Phase 2: Local search from multiple random superincreasing starts
    deadline_phase2 = t0 + DEADLINE_SOFT
    attempts = 0
    while time.time() < deadline_phase2:
        attempts += 1
        # Generate a random superincreasing sequence
        # Use a tighter bound to encourage smaller values
        max_bound = min(best_max, 5000)
        base = _construct_random_superincreasing(rng, max_bound)
        if not _valid(base):
            continue
        base_max = max(base)
        if base_max >= best_max:
            continue

        # Run local search with a time budget
        search_deadline = min(time.time() + 3.0, deadline_phase2)
        result = _local_search(rng, base, search_deadline)
        if result is not None and _valid(result):
            if max(result) < best_max:
                best = result
                best_max = max(best)

        # Also try local search from the current best
        if attempts % 5 == 0:
            search_deadline = min(time.time() + 2.0, deadline_phase2)
            result = _local_search(rng, best, search_deadline)
            if result is not None and _valid(result):
                if max(result) < best_max:
                    best = result
                    best_max = max(best)

        if best_max <= 200:
            break

    # Final safety check
    if time.time() < DEADLINE:
        if not _valid(best):
            best = [1 << i for i in range(N)]

    return best
