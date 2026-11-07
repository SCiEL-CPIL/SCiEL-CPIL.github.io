"""Search-based approach to maximize set size under divisibility constraint on subset sums."""
import time
import random

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 55.0


def _subset_sums_incr(sums_set, x):
    """Given existing subset sums set (nonempty), add x to all and to itself."""
    return sums_set | {s + x for s in sums_set} | {x}


def _divisibility_conflict(sums_set):
    """Check if any two distinct values in sums_set divide one another."""
    if len(sums_set) < 2:
        return False
    vals = sorted(sums_set)
    n = len(vals)
    max_val = vals[-1]
    for i in range(n):
        vi = vals[i]
        if vi > max_val // 2:
            break  # All remaining are also > max/2
        multiple = vi * 2
        while multiple <= max_val:
            if multiple in sums_set:
                return True
            multiple += vi
    return False


def _is_valid_set(elements):
    """Full validation: all nonempty subset sums, no two distinct divide each other."""
    if len(elements) == 0:
        return True
    sums_set = set()
    for x in elements:
        sums_set = _subset_sums_incr(sums_set, x)
    return not _divisibility_conflict(sums_set)


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


def _local_search(elements, pool, rng, max_size=30, time_limit=2.0):
    """Try to improve the set via remove+add and swap operations."""
    if not elements:
        return elements
    
    current = list(elements)
    sums_set = set()
    for x in current:
        sums_set = _subset_sums_incr(sums_set, x)
    
    deadline = time.time() + time_limit
    current_pool = [x for x in pool if x not in set(current)]
    
    improved = True
    while improved and time.time() < deadline and len(current) < max_size:
        improved = False
        rng.shuffle(current_pool)
        
        # Try adding a new element
        for x in current_pool:
            if time.time() > deadline:
                break
            new_sums = _subset_sums_incr(sums_set, x)
            if not _divisibility_conflict(new_sums):
                current.append(x)
                sums_set = new_sums
                current_pool.remove(x)
                improved = True
                break
        
        # Try swap: remove one, add one
        if not improved and len(current) >= 2:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                # Remove from sums
                # Recompute sums without removed
                new_elements = [e for j, e in enumerate(current) if j != i]
                new_sums = set()
                for e in new_elements:
                    new_sums = _subset_sums_incr(new_sums, e)
                
                # Try adding each candidate
                for x in current_pool[:20]:  # Limit candidates for speed
                    if x == removed:
                        continue
                    test_sums = _subset_sums_incr(new_sums, x)
                    if not _divisibility_conflict(test_sums):
                        # Do the swap
                        current[i] = x
                        current_pool.remove(x)
                        if removed not in current_pool:
                            current_pool.append(removed)
                        sums_set = test_sums
                        improved = True
                        break
                if improved:
                    break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    pool_upper = list(range(129, 257))
    pool_full = list(range(1, 257))

    best = []
    best_size = 0

    rng = random.Random(42)

    # Strategy 1: Greedy descending on upper pool
    order_desc = sorted(pool_upper, reverse=True)
    t_remaining = deadline - time.time()
    if t_remaining > 0.5:
        result = _greedy_order_based(pool_upper, order_desc, max_size=30, time_limit=min(t_remaining - 0.1, 3.0))
        if len(result) > best_size:
            best = result
            best_size = len(result)

    # Strategy 2: Multiple randomized restarts with local search
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
        t_remaining = deadline - time.time() - 1.0
        if t_remaining < 0.5:
            break

        # Create shuffled pool with some structure
        if r % 4 == 0:
            pool_shuffled = sorted(pool_upper, reverse=True)
        elif r % 4 == 1:
            pool_shuffled = sorted(pool_upper)
        elif r % 4 == 2:
            pool_shuffled = list(pool_upper)
            rng.shuffle(pool_shuffled)
        else:
            # Mix: start with large, then random
            pool_shuffled = sorted(pool_upper, reverse=True)[:20] + [x for x in pool_upper if x < 149]
            rng.shuffle(pool_shuffled)

        # Greedy build
        t_greedy = min(t_remaining * 0.4, 1.5)
        result = _greedy_order_based(pool_upper, pool_shuffled, max_size=30, time_limit=t_greedy)
        
        # Local search to improve
        if result and len(result) >= 3:
            t_local = min(t_remaining * 0.5, 2.0)
            result = _local_search(result, pool_full, rng, max_size=30, time_limit=t_local)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)

    # Strategy 3: Try extending best with full pool
    if best and time.time() < deadline - 1.0:
        current = list(best)
        sums_set = set()
        for x in current:
            sums_set = _subset_sums_incr(sums_set, x)
        
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

    # Final local search on best
    if best and time.time() < deadline - 0.5:
        t_final = deadline - time.time() - 0.3
        if t_final > 0.5:
            best = _local_search(best, pool_full, rng, max_size=30, time_limit=min(t_final, 3.0))

    # Verify best is valid
    if best:
        if not _is_valid_set(best):
            best = [129, 130, 131]

    result = [int(x) for x in best]
    return result

# EVOLVE-BLOCK-END
