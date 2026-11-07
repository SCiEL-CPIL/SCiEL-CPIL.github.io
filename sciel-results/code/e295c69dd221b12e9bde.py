import time
import random
from math import gcd

DEADLINE_SECONDS = 82.0

def _has_divisibility_conflict(sums):
    """
    Check if any two distinct values in `sums` divide each other.
    Optimized for small sets of integers.
    """
    if len(sums) < 2:
        return False
    
    # Convert to list and sort for efficient bound checking
    arr = sorted(sums)
    n = len(arr)
    
    # If the smallest element a is such that 2*a > max(arr), no conflict possible
    if 2 * arr[0] > arr[-1]:
        return False
        
    # For each a, check if any multiple exists in arr
    # We can use a set for O(1) lookup of multiples
    sum_set = set(arr)
    max_val = arr[-1]
    
    for i in range(n):
        a = arr[i]
        # If 2*a > max_val, no multiples can exist
        if 2 * a > max_val:
            break
        # Check multiples 2a, 3a...
        multiple = 2 * a
        while multiple <= max_val:
            if multiple in sum_set:
                return True
            multiple += a
            
    return False

def _can_add(current_sums_set, current_sums_list, x):
    """
    Check if x can be added to the set defined by current_sums.
    Returns (new_sums_set, new_sums_list) or None.
    """
    # 1. Check distinctness
    # x must not be in current sums
    if x in current_sums_set:
        return None
        
    # New sums are {x} U {s+x for s in current_sums}
    # First check all new sums for collisions with existing sums
    new_sums_added = []
    for s in current_sums_list:
        ns = s + x
        if ns in current_sums_set:
            return None
        new_sums_added.append(ns)
    
    # Check if x collides with any new sum (shouldn't happen since s>0)
    # Check for collisions among new sums themselves
    # s1 + x == s2 + x implies s1 == s2, so no collision among new sums
    
    # Build new sets
    new_sums = set(current_sums_set)
    new_sums.add(x)
    for ns in new_sums_added:
        new_sums.add(ns)
    
    new_sums_list = list(current_sums_list)
    new_sums_list.append(x)
    new_sums_list.extend(new_sums_added)
        
    # 2. Check divisibility
    if _has_divisibility_conflict(new_sums):
        return None
        
    return new_sums, new_sums_list

def _greedy_construct(pool, order_key, start_time, deadline):
    """
    Greedy construction.
    order_key: 'asc', 'desc', 'random', 'prime', etc.
    """
    elements = []
    sums_set = set()
    sums_list = []
    
    # Prepare candidates
    candidates = list(pool)
    if order_key == 'asc':
        candidates.sort()
    elif order_key == 'desc':
        candidates.sort(reverse=True)
    elif order_key == 'random':
        random.shuffle(candidates)
    elif order_key == 'sparse':
        # Prioritize numbers that are less likely to create divisibility conflicts
        # Higher numbers tend to have fewer multiples in range
        candidates.sort(key=lambda x: -x)
    elif order_key == 'prime_bias':
        # Bias toward primes (fewer divisors)
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        candidates.sort(key=lambda x: (not is_prime(x), x))
    else:
        random.shuffle(candidates)
        
    for x in candidates:
        if time.time() > deadline:
            break
        res = _can_add(sums_set, sums_list, x)
        if res is not None:
            sums_set, sums_list = res
            elements.append(x)
            
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_elements = []
    
    # Pool strategies
    # 1. All 1..256
    # 2. 2..256 (exclude 1)
    # 3. 3..256 (exclude 1,2)
    # 4. 5..256 (exclude small numbers)
    # 5. Higher range bias
    pools = [
        list(range(1, 257)),
        list(range(2, 257)),
        list(range(3, 257)),
        list(range(5, 257)),
        list(range(10, 257)),
        list(range(20, 257)),
    ]
    
    orders = ['asc', 'desc', 'random', 'sparse', 'prime_bias']
    
    # Run Greedy for each pool and order
    for pool in pools:
        for order in orders:
            if time.time() > deadline - 10:
                break
            elems = _greedy_construct(pool, order, start_time, deadline)
            if len(elems) > len(best_elements):
                best_elements = elems
                
    # Random Restarts with mixed pools
    max_restarts = 8000
    for _ in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Pick a random pool with bias toward larger ranges
        r = random.random()
        if r < 0.3:
            pool = list(range(1, 257))
        elif r < 0.5:
            pool = list(range(2, 257))
        elif r < 0.7:
            pool = list(range(5, 257))
        elif r < 0.85:
            pool = list(range(10, 257))
        else:
            pool = list(range(20, 257))
        
        # Sometimes use a random subset to encourage diversity
        if random.random() < 0.4:
            pool = random.sample(pool, k=min(40, len(pool)))
            
        # Random order selection
        order = random.choice(['random', 'desc', 'sparse'])
        elems = _greedy_construct(pool, order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            
    # Local Search / Improvement
    # Try to add more elements to the best set found
    if best_elements and time.time() < deadline - 2:
        used = set(best_elements)
        
        # Rebuild sums for best_elements
        current_sums_set = set()
        current_sums_list = []
        for e in best_elements:
            res = _can_add(current_sums_set, current_sums_list, e)
            if res:
                current_sums_set, current_sums_list = res
            else:
                # Should not happen if best_elements is valid
                pass
                
        remaining = [x for x in range(1, 257) if x not in used]
        random.shuffle(remaining)
        
        improved = False
        for x in remaining:
            if time.time() > deadline - 0.5:
                break
            res = _can_add(current_sums_set, current_sums_list, x)
            if res is not None:
                current_sums_set, current_sums_list = res
                best_elements.append(x)
                improved = True
                
        # Second pass with different ordering
        if improved and time.time() < deadline - 1:
            remaining2 = [x for x in range(1, 257) if x not in set(best_elements)]
            random.shuffle(remaining2)
            for x in remaining2:
                if time.time() > deadline - 0.3:
                    break
                res = _can_add(current_sums_set, current_sums_list, x)
                if res is not None:
                    current_sums_set, current_sums_list = res
                    best_elements.append(x)
                    
    return list(best_elements)
