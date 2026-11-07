import time
import random
import sys

DEADLINE_SECONDS = 85.0  # Safety margin for 90s limit

def _has_divisibility_conflict(sums):
    """
    Check if any two distinct values in `sums` divide each other.
    Optimized: Sort sums, for each sum s, check if any multiple 2s, 3s, ... is in the set.
    """
    if len(sums) < 2:
        return False
    
    arr = sorted(sums)
    n = len(arr)
    
    # If the smallest element a is such that 2*a > max(arr), no conflict possible
    if 2 * arr[0] > arr[-1]:
        return False
        
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

def _can_add(current_sums_set, x):
    """
    Check if x can be added to the set defined by current_sums_set.
    Returns (new_sums_set) or None.
    We assume current_sums_set is valid.
    New sums are {x} U {s+x for s in current_sums_set}.
    """
    # 1. Check distinctness of new sums
    # x must not be in current sums
    if x in current_sums_set:
        return None
        
    # New sums added: s + x for s in current_sums_set
    # Check if any s+x is in current_sums_set
    # Also check if any s+x == t+x (impossible if s!=t)
    # Check if x == s+x (impossible since s>0)
    
    new_sums_added = []
    for s in current_sums_set:
        ns = s + x
        if ns in current_sums_set:
            return None
        new_sums_added.append(ns)
    
    # 2. Check divisibility among all new sums
    # We don't need to re-check divisibility among old sums (already valid).
    # We don't need to check divisibility among new sums added (s1+x, s2+x -> (s2-s1) divides 0? No, we check if one divides the other.
    # If (s1+x) divides (s2+x), let s2 = k*s1 + r? No, simple check is safer.
    # However, we can just build the new set and check the whole thing, but that's O(N^2) or O(N log N).
    # Given N is small (< 30 elements -> < 2^30 sums? No, max 30 elements means max 2^30-1 sums which is huge.
    # Wait, the set size is max 30 integers. The number of subset sums is at most 2^30 - 1.
    # But we only generate sums for the *current* set.
    # If we have 7 elements, we have 127 sums.
    # If we have 10 elements, we have 1023 sums.
    # The _has_divisibility_conflict function is O(M log M) where M is number of sums.
    # For M=1000, this is fast.
    
    new_sums = set(current_sums_set)
    new_sums.add(x)
    new_sums.update(new_sums_added)
    
    if _has_divisibility_conflict(new_sums):
        return None
        
    return new_sums

def _greedy_construct(pool, order_key, start_time, deadline):
    """
    Greedy construction.
    """
    elements = []
    sums_set = set()
    
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
        res = _can_add(sums_set, x)
        if res is not None:
            sums_set = res
            elements.append(x)
            
    return elements

def _local_search(elements, start_time, deadline):
    """
    Try to improve the set by swapping elements or adding new ones.
    Returns the best set found.
    """
    if not elements:
        return elements
    
    best_elements = list(elements)
    # Rebuild sums for best_elements
    best_sums = set()
    for e in best_elements:
        res = _can_add(best_sums, e)
        if res is None:
            # Should not happen if elements is valid
            return elements
        best_sums = res
        
    best_size = len(best_elements)
    
    # 1. Try adding more elements
    used = set(best_elements)
    remaining = [x for x in range(1, 257) if x not in used]
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        res = _can_add(best_sums, x)
        if res is not None:
            best_sums = res
            best_elements.append(x)
            best_size += 1
            
    # 2. Try swaps: remove one element, try to add two others? Or just one?
    # Swapping one for one might not increase size.
    # Swapping one for two is better for size.
    # Let's try: for each element e in best_elements, remove it, then try to add 2 new elements.
    # This is expensive, so we limit the number of swap attempts.
    
    # A simpler and effective local search:
    # Randomly pick an element to remove.
    # Try to add random elements to the reduced set.
    # If we can add more than 1 element, we have improved.
    
    max_swap_attempts = 50
    for _ in range(max_swap_attempts):
        if time.time() > deadline - 1:
            break
            
        if len(best_elements) < 2:
            break
            
        # Pick a random element to remove
        rem_idx = random.randint(0, len(best_elements) - 1)
        rem_elem = best_elements[rem_idx]
        
        # Create a new list without rem_elem
        trial_elements = best_elements[:rem_idx] + best_elements[rem_idx+1:]
        
        # Rebuild sums for trial_elements
        trial_sums = set()
        valid = True
        for e in trial_elements:
            res = _can_add(trial_sums, e)
            if res is None:
                valid = False
                break
            trial_sums = res
            
        if not valid:
            continue
            
        # Try to add new elements to trial_elements
        used_trial = set(trial_elements)
        candidates = [x for x in range(1, 257) if x not in used_trial]
        random.shuffle(candidates)
        
        added_count = 0
        for x in candidates:
            if time.time() > deadline - 1:
                break
            res = _can_add(trial_sums, x)
            if res is not None:
                trial_sums = res
                trial_elements.append(x)
                added_count += 1
                # If we added more than 1, we definitely improved size
                if added_count > 1:
                    break
                
        # If we added more elements than we removed (1), we have a larger set
        if len(trial_elements) > len(best_elements):
            best_elements = trial_elements
            best_sums = trial_sums
            best_size = len(best_elements)
            # Reset attempts to continue improving
            max_swap_attempts = 50 

    return best_elements

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_elements = []
    
    # Pool strategies
    pools = [
        list(range(1, 257)),
        list(range(2, 257)),
        list(range(3, 257)),
        list(range(5, 257)),
        list(range(10, 257)),
        list(range(20, 257)),
        list(range(50, 257)),
        list(range(100, 257)),
    ]
    
    orders = ['asc', 'desc', 'random', 'sparse', 'prime_bias']
    
    # Phase 1: Greedy Construction
    for pool in pools:
        for order in orders:
            if time.time() > deadline - 20:
                break
            elems = _greedy_construct(pool, order, start_time, deadline)
            if len(elems) > len(best_elements):
                best_elements = elems
                
    # Phase 2: Random Restarts with Local Search
    max_restarts = 100
    for _ in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Pick a random pool
        pool = random.choice(pools)
        # Sometimes use a random subset
        if random.random() < 0.5:
            pool = random.sample(pool, k=min(50, len(pool)))
            
        order = random.choice(['random', 'desc', 'sparse'])
        elems = _greedy_construct(pool, order, start_time, deadline)
        
        # Apply local search to the result
        if len(elems) > 0:
            improved_elems = _local_search(elems, start_time, deadline)
            if len(improved_elems) > len(best_elements):
                best_elements = improved_elems
                
        # Also try to improve the current best directly
        if len(best_elements) > 0 and random.random() < 0.3:
            improved_best = _local_search(best_elements, start_time, deadline)
            if len(improved_best) > len(best_elements):
                best_elements = improved_best

    # Final check: try to add any remaining elements to the best set
    if best_elements and time.time() < deadline - 1:
        used = set(best_elements)
        current_sums = set()
        for e in best_elements:
            res = _can_add(current_sums, e)
            if res:
                current_sums = res
        
        remaining = [x for x in range(1, 257) if x not in used]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 0.5:
                break
            res = _can_add(current_sums, x)
            if res is not None:
                current_sums = res
                best_elements.append(x)
                
    # Ensure return type is list of ints
    return [int(x) for x in best_elements]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
