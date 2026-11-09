"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

START_TIME = time.time()
DEADLINE = 50.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_cross_cardinality(sums, n):
    """Check that for all i != j, sums[i] & sums[j] == 0."""
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def try_add(x, sums, current_size):
    """Try adding element x to the set represented by sums.
    Returns new_sums if valid, else None."""
    new_sums = list(sums)
    for k in range(current_size, -1, -1):
        shifted = sums[k] << x
        # Check cross-cardinality collision: new_sums[k+1] must not collide with any other new_sums[m] where m != k+1
        # But we also need to check against existing sums that don't change
        # Actually, only sums[k+1] changes. Check new_sums[k+1] against all other new_sums[m] for m != k+1
        candidate = new_sums[k+1] | shifted
        for m in range(current_size + 2):
            if m != k + 1 and (candidate & new_sums[m]):
                return None
        new_sums[k+1] = candidate
    return new_sums

def greedy_construct(order):
    """Greedy: iterate through 'order', add each element if valid."""
    sums = [0] * (len(order) + 1)
    sums[0] = 1  # empty subset sum 0
    current_size = 0
    result = []
    for x in order:
        if is_done():
            break
        new_sums = try_add(x, sums, current_size)
        if new_sums is not None:
            sums = new_sums
            current_size += 1
            result.append(x)
    return result

def construct():
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Powers of 2 baseline (9 elements)
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Increasing order
    strategies.append(list(range(1, 257)))
    
    # Strategy 2: Decreasing order
    strategies.append(list(range(256, 0, -1)))
    
    # Strategy 3: Random orders (multiple)
    rng = random.Random(42)
    for _ in range(10):
        order = list(range(1, 257))
        rng.shuffle(order)
        strategies.append(order)
    
    for order in strategies:
        if is_done():
            break
        result = greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
    
    # Ensure Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
