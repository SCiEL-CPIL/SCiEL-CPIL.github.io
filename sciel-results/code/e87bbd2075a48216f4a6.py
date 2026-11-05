import time
import random
from itertools import combinations

def is_valid_incremental(existing_sorted, new_elem):
    """
    Check if adding new_elem to existing_sorted (which is valid) maintains validity.
    existing_sorted must be a sorted list of integers.
    Only need to check:
    1. new_elem doesn't divide sum of any 3 existing elements
    2. No existing element divides sum of new_elem + 2 existing elements
    """
    n = len(existing_sorted)
    
    # Check 1: new_elem vs triples from existing_set
    a = new_elem
    if n >= 3:
        max_sum3 = existing_sorted[-1] + existing_sorted[-2] + existing_sorted[-3]
        if a <= max_sum3:
            min_sum3 = existing_sorted[0] + existing_sorted[1] + existing_sorted[2]
            start_mult = (min_sum3 + a - 1) // a
            end_mult = max_sum3 // a
            if start_mult <= end_mult:
                existing_set_s = set(existing_sorted)
                for k in range(start_mult, end_mult + 1):
                    target = k * a
                    found = False
                    # Iterate pairs in existing_sorted to see if they sum to target - x
                    # We need x + y + z = target
                    # Iterate x, then check if y+z = target-x has a solution
                    for i in range(n - 2):
                        xi = existing_sorted[i]
                        # Optimization: if xi + next two smallest > target, break
                        if xi + existing_sorted[i+1] + existing_sorted[i+2] > target:
                            break
                        rem_target = target - xi
                        # Now find y, z in existing_sorted[i+1:] such that y+z = rem_target
                        # Since existing_sorted is sorted, we can use two pointers or set lookup
                        # But we must ensure y, z are distinct and > xi (to avoid duplicates in iteration? No, just distinct indices)
                        # Actually, since we iterate i, j, k with i<j<k, we just need y > xi and z > y.
                        # So we look in the sublist existing_sorted[i+1:]
                        # Using set for O(1) lookup is faster than scanning if n is large, but we need to ensure distinctness.
                        # Let's use a set of the remaining elements.
                        # However, creating a set for each i is O(n^2). 
                        # Better: just loop j.
                        for j in range(i + 1, n - 1):
                            xj = existing_sorted[j]
                            if xi + xj + existing_sorted[j+1] > target:
                                break
                            rem = target - xi - xj
                            # rem must be in existing_sorted[j+1:]
                            # Since sorted, if rem > xj, we can check if rem is in the set of remaining
                            # To avoid O(n) scan, we precompute a set of the whole list, but we must ensure index > j.
                            # Simplest: if rem > xj and rem in existing_set_s, we still need to ensure rem is not xi or xj.
                            # Since rem > xj > xi, distinctness from xi, xj is guaranteed if rem != xj (which is true since rem > xj) and rem != xi (true).
                            # But we need rem to be in the list at an index > j.
                            # If we just check `rem in existing_set_s`, it might be at an index <= j.
                            # Example: S=[1, 2, 3, 4]. target=6. xi=1, xj=2. rem=3. 3 is in S. Index of 3 is 2. j=1. 2>1. OK.
                            # Example: S=[1, 3, 2, 4] (not sorted). But S is sorted.
                            # If S is sorted, and rem > xj, then the index of rem MUST be > j?
                            # Yes, because if rem > xj, and S is sorted, the first occurrence of rem is after the last occurrence of xj.
                            # Since elements are distinct, index(rem) > index(xj) = j.
                            # So `rem in existing_set_s` is sufficient if rem > xj.
                            if rem > xj and rem in existing_set_s:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
    
    # Check 2: Each existing element vs triples including new_elem
    for idx_a, a in enumerate(existing_sorted):
        # Triples that include new_elem: new_elem + 2 from existing_set (excluding a)
        # Need to check if a divides new_elem + x + y for some x, y in existing_set, x != y, x != a, y != a
        others_excluding_a = [x for x in existing_sorted if x != a]
        m = len(others_excluding_a)
        if m < 2:
            continue
        
        # We need: a | (new_elem + x + y) for some distinct x, y in others_excluding_a
        # Equivalently: (new_elem + x + y) % a == 0
        # i.e., x + y ≡ -new_elem (mod a)
        
        # If a is large, few multiples. If a is small, more to check.
        # Max possible sum: new_elem + others_excluding_a[-1] + others_excluding_a[-2]
        max_sum = new_elem + others_excluding_a[-1] + others_excluding_a[-2]
        if a > max_sum:
            continue
        
        min_sum = new_elem + others_excluding_a[0] + others_excluding_a[1]
        
        start_mult = (min_sum + a - 1) // a
        end_mult = max_sum // a
        
        if start_mult > end_mult:
            continue
        
        # For each multiple target = k*a, check if x + y = target - new_elem for distinct x,y in others_excluding_a
        others_set_s = set(others_excluding_a)
        for k in range(start_mult, end_mult + 1):
            target = k * a
            need = target - new_elem
            if need < 0:
                continue
            # Check if any pair in others_excluding_a sums to need
            found = False
            for i in range(m - 1):
                xi = others_excluding_a[i]
                if xi + others_excluding_a[i+1] > need:
                    break
                rem = need - xi
                if rem > xi and rem in others_set_s:
                    found = True
                    break
            if found:
                return False
    
    return True

def greedy_construct(order_type="descending"):
    """
    Greedy construction.
    order_type: 'descending', 'ascending', 'primes_first'
    """
    selected = []
    selected_set = set()
    
    if order_type == "descending":
        candidates = list(range(192, 0, -1))
    elif order_type == "ascending":
        candidates = list(range(1, 193))
    elif order_type == "primes_first":
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        
        primes = [x for x in range(192, 0, -1) if is_prime(x)]
        composites = [x for x in range(192, 0, -1) if not is_prime(x)]
        candidates = primes + composites
    
    for x in candidates:
        if x in selected_set:
            continue
        # Check validity
        if is_valid_incremental(selected, x):
            selected.append(x)
            selected_set.add(x)
            
    return selected

def local_search(s, rng, deadline):
    """
    Local search to improve the set.
    """
    s = set(s)
    cur_size = len(s)
    stall_count = 0
    
    while time.time() < deadline:
        if stall_count > 100:
            break
            
        # Try to add a random element
        candidates = [x for x in range(1, 193) if x not in s]
        if not candidates:
            break
            
        x = rng.choice(candidates)
        
        # Check if adding x is valid
        # We need a fast check. Since s is small-ish, we can just check.
        # But is_valid_incremental expects a sorted list.
        sorted_s = sorted(s)
        if is_valid_incremental(sorted_s, x):
            s.add(x)
            cur_size += 1
            stall_count = 0
        else:
            stall_count += 1
            
        # Try to swap: remove one, add two
        # This is complex and slow. Let's stick to adding first.
        # If we get stuck, try removing a random element and adding a random one.
        if stall_count > 50 and len(s) > 0:
            # Remove a random element
            to_remove = rng.choice(list(s))
            s.remove(to_remove)
            cur_size -= 1
            stall_count = 0
            
    return list(s)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    
    best = []
    best_size = 0
    rng = random.Random(42)
    
    # Try different greedy orders
    for order in ["descending", "ascending", "primes_first"]:
        res = greedy_construct(order)
        if len(res) > best_size:
            best = res
            best_size = len(res)
            
    # Local search on the best result
    if best_size > 0:
        improved = local_search(best, rng, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            
    # Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline - 5:
        # Random subset start? Or just random greedy?
        # Let's try a random permutation greedy
        candidates = list(range(1, 193))
        rng.shuffle(candidates)
        res = greedy_construct_custom(candidates)
        if len(res) > best_size:
            best = res
            best_size = len(res)
        
        restarts += 1
        if restarts > 50:
            break
            
    return best

def greedy_construct_custom(candidates):
    """
    Greedy construction with a custom candidate order.
    """
    selected = []
    selected_set = set()
    for x in candidates:
        if x in selected_set:
            continue
        if is_valid_incremental(selected, x):
            selected.append(x)
            selected_set.add(x)
    return selected

# Note: The original `construct` in the seed program was a single greedy pass.
# The inspiration suggests multiple strategies and local search.
# The `greedy_construct` in the inspiration used `is_valid_incremental` which is faster.
# I will ensure `construct` returns a list of integers.

# One detail: `is_valid_incremental` in the inspiration assumes `existing_sorted` is sorted.
# In my `greedy_construct`, `selected` is appended in the order of candidates.
# If candidates are descending, `selected` is descending? No, `is_valid_incremental` sorts it?
# No, `is_valid_incremental` takes `existing_sorted`.
# So I must pass `sorted(selected)` to `is_valid_incremental`.
# My `greedy_construct` does NOT sort `selected` before passing.
# I need to fix that.

def is_valid_incremental_corrected(existing_list, new_elem):
    """
    Takes a list (not necessarily sorted) and a new element.
    """
    existing_sorted = sorted(existing_list)
    return is_valid_incremental(existing_sorted, new_elem)

def greedy_construct_fixed(order_type="descending"):
    selected = []
    selected_set = set()
    
    if order_type == "descending":
        candidates = list(range(192, 0, -1))
    elif order_type == "ascending":
        candidates = list(range(1, 193))
    elif order_type == "primes_first":
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        primes = [x for x in range(192, 0, -1) if is_prime(x)]
        composites = [x for x in range(192, 0, -1) if not is_prime(x)]
        candidates = primes + composites
    else:
        candidates = order_type # assume it's a list
        
    for x in candidates:
        if x in selected_set:
            continue
        # Pass sorted list for efficiency inside the check
        if is_valid_incremental(sorted(selected), x):
            selected.append(x)
            selected_set.add(x)
            
    return selected

def local_search_fixed(s, rng, deadline):
    s = set(s)
    cur_size = len(s)
    stall_count = 0
    
    while time.time() < deadline:
        if stall_count > 100:
            break
            
        candidates = [x for x in range(1, 193) if x not in s]
        if not candidates:
            break
            
        x = rng.choice(candidates)
        
        if is_valid_incremental(sorted(list(s)), x):
            s.add(x)
            cur_size += 1
            stall_count = 0
        else:
            stall_count += 1
            
        if stall_count > 50 and len(s) > 0:
            to_remove = rng.choice(list(s))
            s.remove(to_remove)
            cur_size -= 1
            stall_count = 0
            
    return list(s)

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best = []
    best_size = 0
    rng = random.Random(42)
    
    # 1. Standard Greedy Strategies
    for order in ["descending", "ascending", "primes_first"]:
        res = greedy_construct_fixed(order)
        if len(res) > best_size:
            best = res
            best_size = len(res)
            
    # 2. Local Search on Best
    if best_size > 0:
        improved = local_search_fixed(best, rng, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            
    # 3. Random Restarts
    while time.time() < deadline - 5:
        candidates = list(range(1, 193))
        rng.shuffle(candidates)
        res = greedy_construct_fixed(candidates)
        if len(res) > best_size:
            best = res
            best_size = len(res)
            
        # Optional: Local search on this result
        if time.time() < deadline - 10:
            improved = local_search_fixed(res, rng, deadline)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                
    return best
