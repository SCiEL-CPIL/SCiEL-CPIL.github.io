"""Model-and-optimize approach for divisibility-incomparable subset sums.

Strategy:
- The constraint is that no two distinct nonempty subset sums divide one another.
- This is a very strong constraint. If we have numbers a < b, then a + b must not be divisible by any other subset sum, and vice versa.
- In particular, if we have two subsets S1 and S2 with sums s1 < s2, we require s2 % s1 != 0.
- Also, all subset sums must be distinct (otherwise equal sums would lead to a division issue as described in the task).

Approach:
1. Fast subset sum computation using iterative doubling.
2. Efficient divisibility check: for each sum s, check if any multiple of s (greater than s) is in the set.
3. Greedy construction with multiple random orders.
4. Simulated annealing with swap and add/remove moves.
5. Return the best set found within the time budget.
"""

import time
import random

START_TIME = time.time()
TIME_LIMIT = 85.0  # 90s kill, leave 5s margin

def compute_all_sums(nums):
    """Generate all nonempty subset sums of nums as a set using iterative doubling."""
    sums = set()
    for num in nums:
        # Add num to all existing sums
        new_sums = set()
        for s in sums:
            new_sums.add(s + num)
        sums = sums | new_sums | {num}
    return sums

def check_divisibility(sums_set):
    """Check if any two distinct sums in the set have a divisibility relation.
    
    For each sum s, check if any multiple 2s, 3s, ... is also in the set.
    This is O(n * max_sum/s) which is much faster than O(n^2) for typical cases.
    """
    if not sums_set:
        return True
    
    max_sum = max(sums_set)
    
    for s in sums_set:
        # Check multiples of s that could be in the set
        for k in range(2, max_sum // s + 1):
            if s * k in sums_set:
                return False
    
    return True

def is_valid_set(nums):
    """Check if a set of numbers satisfies the constraint."""
    if not nums:
        return True
    sums = compute_all_sums(nums)
    return check_divisibility(sums)

def can_add(nums, new_elem):
    """Check if adding new_elem to nums keeps the set valid.
    
    Incremental: compute existing sums, then new sums, and check divisibility.
    """
    if not nums:
        return True  # Single element always valid
    
    # Get existing subset sums
    existing_sums = compute_all_sums(nums)
    
    # Build the full new sum set
    new_sums = set(existing_sums)
    for s in existing_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    # Check divisibility in new_sums
    return check_divisibility(new_sums)

def greedy_construct(pool, order_seed=0):
    """Greedy construction: try adding elements in a given order."""
    rng = random.Random(order_seed)
    result = []
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for elem in remaining:
        if can_add(result, elem):
            result.append(elem)
            if len(result) >= 30:
                break
        if time.time() - START_TIME > TIME_LIMIT:
            break
    return result

def simulated_annealing(nums, pool, seed=0, max_iters=2000):
    """Simulated annealing to improve the set.
    
    Moves: swap (replace one element), add (if space), remove (to try different combos).
    Accepts worse solutions with probability exp(-delta/T) where delta = size decrease.
    """
    rng = random.Random(seed)
    current = list(nums)
    current_set = set(current)
    available = [x for x in pool if x not in current_set]
    
    best = list(current)
    best_size = len(current)
    
    T = 1.0
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        if T < 0.001:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        trial = None
        
        if move_type < 0.5 and len(current) > 0 and available:
            # Swap: remove one, add another
            i = rng.randrange(len(current))
            j = rng.randrange(len(available))
            trial = current[:i] + current[i+1:] + [available[j]]
        
        elif move_type < 0.75 and available and len(current) < 30:
            # Add: try to add a random available element
            j = rng.randrange(len(available))
            trial = current + [available[j]]
        
        elif move_type < 0.9 and len(current) > 1:
            # Remove: try removing a random element (to explore)
            i = rng.randrange(len(current))
            trial = current[:i] + current[i+1:]
        
        else:
            # Swap with random pair
            if len(current) >= 2 and available:
                i1, i2 = rng.sample(range(len(current)), 2)
                j = rng.randrange(len(available))
                trial = current[:i1] + current[i1+1:i2] + current[i2+1:] + [available[j]]
        
        if trial is None:
            T *= cooling
            continue
        
        # Check validity
        if is_valid_set(trial):
            delta = len(trial) - len(current)
            if delta >= 0:
                # Accept (same or better)
                current = trial
                current_set = set(current)
                available = [x for x in pool if x not in current_set]
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    if best_size >= 30:
                        break
            else:
                # Worse: accept with probability exp(delta/T)
                if rng.random() < (0.01 ** (delta / T)):
                    current = trial
                    current_set = set(current)
                    available = [x for x in pool if x not in current_set]
        
        T *= cooling
    
    return best

def construct():
    """Main entry point. Returns a list of 1..30 distinct integers in 1..256."""
    # Pool: all integers in 1..256
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy constructions with different biases
    for attempt in range(15):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Vary the pool ordering strategy
        if attempt % 3 == 0:
            # Random order
            result = greedy_construct(pool, order_seed=attempt)
        elif attempt % 3 == 1:
            # Descending (larger numbers first)
            pool_desc = list(range(256, 0, -1))
            result = greedy_construct(pool_desc, order_seed=attempt + 100)
        else:
            # Medium range bias
            pool_med = list(range(30, 220))
            result = greedy_construct(pool_med, order_seed=attempt + 200)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= 30:
                break
    
    # Phase 2: Simulated annealing on the best solutions
    if best_size < 30 and time.time() - START_TIME < TIME_LIMIT - 5:
        # Run SA on the best solution
        improved = simulated_annealing(best, pool, seed=42, max_iters=3000)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: More SA attempts with different seeds if time allows
    if best_size < 30 and time.time() - START_TIME < TIME_LIMIT - 3:
        for sa_seed in [1, 2, 3, 7, 13, 42, 99, 123]:
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
            
            # Start from a fresh greedy or the current best
            if sa_seed % 2 == 0:
                start = greedy_construct(pool, order_seed=sa_seed * 100)
            else:
                start = list(best)
            
            improved = simulated_annealing(start, pool, seed=sa_seed, max_iters=1500)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                if best_size >= 30:
                    break
    
    # If we didn't find anything good, return a known valid small set
    if not best:
        best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
