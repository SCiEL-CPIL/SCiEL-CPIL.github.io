import time
import random
from math import gcd

DEADLINE_SECONDS = 85.0

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
    
    for i in range(n):
        a = arr[i]
        # Check multiples 2a, 3a...
        multiple = 2 * a
        # We can break if multiple > max(arr)
        while multiple <= arr[-1]:
            if multiple in sum_set:
                return True
            multiple += a
        # Optimization: if a is large, few multiples to check
        # If 2*a > arr[-1], break (already handled by initial check, but good for inner loop)
        if 2 * a > arr[-1]:
            break
            
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
    # Check if any s+x is in current_sums
    # Check if any s+x == x (implies s=0, impossible)
    # Check if any s1+x == s2+x (implies s1=s2, impossible)
    
    new_sums = set(current_sums_set)
    new_sums_list = list(current_sums_list)
    
    # Add x
    new_sums.add(x)
    new_sums_list.append(x)
    
    # Add s+x
    for s in current_sums_list:
        ns = s + x
        if ns in current_sums_set:
            # Collision with old sum
            return None
        if ns in new_sums:
            # Collision with another new sum (s1+x == s2+x or s1+x == x)
            # s1+x == x => s1=0 (no)
            # s1+x == s2+x => s1=s2 (no)
            # So this should only happen if ns was added twice? 
            # No, we are iterating unique s.
            # But ns could equal x? No, s>0.
            # So ns is distinct from x.
            # And ns is distinct from other ns' because s are distinct.
            # So the only collision is with current_sums_set.
            return None
        new_sums.add(ns)
        new_sums_list.append(ns)
        
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
        # Prioritize numbers with large gaps? 
        # Just random for now, but maybe weighted by value?
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
    # 4. Random subsets
    
    pools = [
        list(range(1, 257)),
        list(range(2, 257)),
        list(range(3, 257)),
        list(range(5, 257)),
    ]
    
    orders = ['asc', 'desc', 'random']
    
    # Run Greedy for each pool and order
    for pool in pools:
        for order in orders:
            if time.time() > deadline - 5:
                break
            elems = _greedy_construct(pool, order, start_time, deadline)
            if len(elems) > len(best_elements):
                best_elements = elems
                
    # Random Restarts with mixed pools
    max_restarts = 5000
    for _ in range(max_restarts):
        if time.time() > deadline - 2:
            break
            
        # Pick a random pool
        pool = random.choice(pools)
        # Sometimes use a random subset to encourage diversity
        if random.random() < 0.5:
            pool = random.sample(pool, k=min(50, len(pool)))
            
        elems = _greedy_construct(pool, 'random', start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            
    # Local Search / Improvement
    # Try to add more elements to the best set found
    if best_elements and time.time() < deadline - 1:
        used = set(best_elements)
        sums_set, sums_list = None, None
        
        # Reconstruct sums for best_elements
        # This is O(N * 2^N) if done naively, but we can rebuild from scratch
        # Or just use the sums from the last successful greedy? 
        # We didn't store them. Let's rebuild.
        
        # Rebuild sums
        s_set = set()
        s_list = []
        # Efficient rebuild:
        # Start with empty
        # Add each element
        # This is O(N * |S|)
        
        # Actually, we can just try to add remaining elements to the best set
        # We need the sums of the best set.
        
        # Let's just do a quick greedy pass on the remaining elements
        # using the sums of the current best set.
        
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
                
    return list(best_elements)
