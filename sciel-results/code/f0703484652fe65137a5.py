import time
import random
import sys

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def try_add(x, sums, current_size):
    """Try adding element x to the set represented by sums (list of Python ints as bitsets).
    Returns new_sums if valid, else None."""
    new_sums = list(sums)
    for k in range(current_size, -1, -1):
        shifted = sums[k] << x
        candidate = new_sums[k + 1] | shifted
        # Check cross-cardinality collision
        for m in range(current_size + 2):
            if m != k + 1 and (candidate & new_sums[m]):
                return None
        new_sums[k + 1] = candidate
    return new_sums

def greedy_construct(order):
    """Greedy: iterate through 'order', add each element if valid.
    Then do a second pass over all remaining candidates."""
    sums = [0] * (len(order) + 1)
    sums[0] = 1  # empty subset sum 0
    current_size = 0
    result = []
    used = set()
    
    # First pass
    for x in order:
        if is_done():
            break
        new_sums = try_add(x, sums, current_size)
        if new_sums is not None:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    # Second pass: try all remaining candidates
    remaining = [x for x in order if x not in used]
    for x in remaining:
        if is_done():
            break
        new_sums = try_add(x, sums, current_size)
        if new_sums is not None:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    return result

def construct():
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Powers of 2 baseline (9 elements)
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Increasing order
    strategies.append(list(range(1, 257)))
    
    # Strategy 2: Decreasing order
    strategies.append(list(range(256, 0, -1)))
    
    # Strategy 3: Random orders (more of them)
    rng = random.Random(42)
    for _ in range(30):
        order = list(range(1, 257))
        rng.shuffle(order)
        strategies.append(order)
    
    # Strategy 4: Sorted by value, but with some structure
    # Try multiples of small primes first
    for prime in [2, 3, 5, 7, 11]:
        order = [x for x in range(1, 257) if x % prime == 0]
        order += [x for x in range(1, 257) if x % prime != 0]
        strategies.append(order)
    
    for order in strategies:
        if is_done():
            break
        result = greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
    
    # Ensure Python ints in a list
    return [int(x) for x in best_set]
