"""Search-based approach to maximize set size under divisibility constraint on subset sums."""
import time
import random

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 50.0


def _subset_sums_incr(sums_set, x):
    """Given existing subset sums set (nonempty), add x to all and to itself.
    Returns the new set of subset sums for the augmented list."""
    return sums_set | {s + x for s in sums_set} | {x}


def _divisibility_conflict(sums_set):
    """Check if any two distinct values in sums_set divide one another.
    Returns True if conflict exists (invalid), False otherwise."""
    if len(sums_set) < 2:
        return False
    vals = sorted(sums_set)
    n = len(vals)
    max_val = vals[-1]
    for i in range(n):
        vi = vals[i]
        if vi > max_val // 2:
            continue
        multiple = vi * 2
        while multiple <= max_val:
            if multiple in sums_set:
                return True
            multiple += vi
    return False


def _is_valid_set(elements):
    """Full validation: all nonempty subset sums, no two distinct divide each other."""
    n = len(elements)
    if n == 0:
        return True
    sums_set = set()
    for x in elements:
        sums_set = _subset_sums_incr(sums_set, x)
    return not _divisibility_conflict(sums_set)


def _greedy_randomized(elements_pool, rng, max_size=30, time_limit=None):
    """Randomized greedy: repeatedly pick a random element that keeps validity."""
    current = []
    sums_set = set()
    remaining = list(elements_pool)
    rng.shuffle(remaining)

    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = None

    while len(current) < max_size and remaining:
        found = False
        for i in range(len(remaining)):
            x = remaining[i]
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                current.append(x)
                sums_set = new_sums
                remaining.pop(i)
                found = True
                break
            if deadline is not None and time.time() > deadline:
                break
        if not found:
            break
        if deadline is not None and time.time() > deadline:
            break

    return current


def _greedy_order_based(elements_pool, order, max_size=30, time_limit=None):
    """Greedy: try elements in given order, add if valid."""
    current = []
    sums_set = set()

    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = None

    for x in order:
        if len(current) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        new_sums = _subset_sums_incr(sums_set, x)
        if not _divisibility_conflict(new_sums):
            current.append(x)
            sums_set = new_sums

    return current


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Candidate pool: focus on upper half to reduce divisibility risks
    # Numbers in [129, 256] are less likely to divide sums of other numbers
    # because their smallest multiple is >= 258, which is close to the max sum of two small numbers
    # but we are picking large numbers, so sums are large.
    # Actually, if we pick from [129, 256], min sum is 129.
    # 2 * 129 = 258.
    # If we avoid sums being exactly 258, 387, etc., we are safe from 129.
    
    pool_upper = list(range(129, 257))
    pool_full = list(range(1, 257))

    best = []
    best_size = 0

    rng = random.Random(42)

    # Strategy 1: Greedy on upper pool, descending (largest first)
    # Largest first might help because they are "safe" divisors?
    # No, large numbers are safe as DIVIDENDS (hard to be divided by small ones if small ones aren't there).
    # But they are dangerous as DIVISORS? No, large divisors have few multiples.
    # So large numbers are generally safer.
    
    order_desc = sorted(pool_upper, reverse=True)
    t_remaining = deadline - time.time()
    if t_remaining > 0.5:
        result = _greedy_order_based(pool_upper, order_desc, max_size=30, time_limit=min(t_remaining - 0.1, 5.0))
        if len(result) > best_size:
            best = result
            best_size = len(result)

    # Strategy 2: Randomized greedy on upper pool
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
        t_remaining = deadline - time.time() - 1.0
        if t_remaining < 0.2:
            break

        # Shuffle upper pool
        pool_shuffled = list(pool_upper)
        rng.shuffle(pool_shuffled)
        
        # Sometimes try ascending?
        if r % 5 == 0:
            pool_shuffled.sort()
        elif r % 5 == 1:
            pool_shuffled.sort(reverse=True)
        # else keep shuffled

        result = _greedy_randomized(pool_shuffled, rng, max_size=30, time_limit=min(t_remaining - 0.5, 1.0))
        if len(result) > best_size:
            best = result
            best_size = len(result)

    # Strategy 3: Try to extend best set with full pool (including small numbers)
    # Small numbers are risky, but if we have a large set of large numbers,
    # maybe a small number fits? Unlikely, but check.
    if best and time.time() < deadline - 1.0:
        current = list(best)
        sums_set = set()
        for x in current:
            sums_set = _subset_sums_incr(sums_set, x)
        
        # Check full pool in random order
        remaining_pool = [x for x in pool_full if x not in set(best)]
        rng.shuffle(remaining_pool)
        
        for x in remaining_pool:
            if time.time() > deadline - 0.5:
                break
            if len(current) >= 30:
                break
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                current.append(x)
                sums_set = new_sums
        
        if len(current) > best_size:
            best = current
            best_size = len(best)

    # Verify best is valid
    if best:
        if not _is_valid_set(best):
            best = [129, 130, 131]

    result = [int(x) for x in best]
    return result

# EVOLVE-BLOCK-END
