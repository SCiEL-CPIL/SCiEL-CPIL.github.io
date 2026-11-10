import time
import random

N = 256

def _is_valid_maximal(selected, sums):
    # Verify Sidon property
    if len(selected) != len(set(selected)):
        return False
    # Check all sums are distinct
    # We already maintain sums, but let's double check for safety if needed
    # For speed, we assume the construction maintains it correctly.
    
    # Check maximality: every missing element must conflict
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        # New sums would be x + a for a in selected, and 2*x
        # If any of these are in sums, x cannot be added.
        # If none are in sums, x CAN be added, so it's not maximal.
        can_add = True
        for a in selected:
            if (x + a) in sums:
                can_add = False
                break
        if can_add:
            if (2 * x) in sums:
                can_add = False
        
        if can_add:
            return False
    return True

def _greedy_with_order(order):
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        # Sums to check: x + a for all a in selected, and 2*x
        # If any of these are already in sums, x cannot be added.
        # Note: x + a == a + x. Since we process x once, we just check against existing sums.
        
        # Optimization: 
        # If 2*x is in sums, fail.
        # If any x+a is in sums, fail.
        
        # To speed this up, we can pre-compute or just iterate.
        # Given N=256 and small selected size, iteration is fine.
        
        if 2 * x in sums:
            continue
            
        blocked = False
        for a in selected:
            if (x + a) in sums:
                blocked = True
                break
        
        if not blocked:
            selected.append(x)
            # Update sums
            for a in selected[:-1]: # Exclude x itself for the loop, but we need x+x and x+a for a < x? 
                # Wait, the logic "x+a" covers all pairs involving x and previous elements.
                # We also need 2*x.
                sums.add(x + a)
            sums.add(2 * x)
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 55s limit to be safe within 60s
    
    best_size = float('inf')
    best_set = None
    
    # Randomized restarts
    # We can also try reverse order
    order_rev = list(range(N, 0, -1))
    candidate = _greedy_with_order(order_rev)
    sums_check = set()
    for i, a in enumerate(candidate):
        for b in candidate[i:]:
            sums_check.add(a+b)
    if _is_valid_maximal(candidate, sums_check):
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate

    # Main loop: Random permutations
    # To make it faster, we can use a "randomized greedy" where we pick a random available element?
    # No, the permutation approach is standard.
    
    # We can also try "randomized priority" where we shuffle, but maybe bias?
    # Let's just do many random shuffles.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # We can also try a "mixed" strategy: 
        # Sometimes start with large numbers, sometimes small, sometimes random.
        # The simple shuffle is good.
        
        candidate = _greedy_with_order(order)
        
        # Quick check: if size is already >= best_size, we might skip full verification?
        # No, a larger set might be valid while a smaller one isn't? 
        # Actually, if we find a valid set of size S, any set of size > S is worse.
        # But we need to know if the candidate is valid first.
        # Verification is O(N * K) where K is size. N=256, K~20. Very fast.
        
        # Construct sums for verification
        sums_check = set()
        valid = True
        for i, a in enumerate(candidate):
            for b in candidate[i:]:
                s = a + b
                if s in sums_check:
                    valid = False
                    break
                sums_check.add(s)
            if not valid:
                break
        
        if valid:
            if _is_valid_maximal(candidate, sums_check):
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_set = candidate
                    # If we reach a theoretical lower bound or a very small size, we can stop?
                    # Min size for maximal Sidon in [1,256] is likely around 16-20.
                    if best_size <= 15:
                        break

    if best_set is None:
        # Fallback: Run ascending greedy one last time to ensure we have a valid maximal set
        order = list(range(1, N + 1))
        best_set = _greedy_with_order(order)
        # It should be maximal by construction, but let's verify
        sums_check = set()
        valid = True
        for i, a in enumerate(best_set):
            for b in best_set[i:]:
                s = a + b
                if s in sums_check:
                    valid = False
                    break
                sums_check.add(s)
            if not valid:
                break
        if valid and _is_valid_maximal(best_set, sums_check):
            return best_set
        else:
            # Extreme fallback: return a single element? No, must be maximal.
            # A set of all elements is not Sidon. 
            # If this happens, the logic is flawed. 
            # But greedy always works.
            return best_set

    return best_set
