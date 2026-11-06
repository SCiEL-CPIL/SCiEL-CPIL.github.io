import random
import time

def construct():
    n = 512
    
    def creates_4ap_with_x(S, x):
        # S is a set
        # Check if adding x to S creates a 4-AP involving x.
        
        # Case 1: x is the first term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        # Case 2: x is the second term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # Case 3: x is the third term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        # Case 4: x is the fourth term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
        return False

    def valid_swap(S, x_out, x_in):
        # Check if (S - {x_out}) U {x_in} is 4-AP free.
        # We don't need to check the whole set, only 4-APs involving x_in.
        # Because removing x_out cannot create a new 4-AP.
        # So we just need to check if x_in creates a 4-AP in (S - {x_out}).
        # Let S_prime = S - {x_out}.
        # We can simulate this by temporarily removing x_out from S, checking x_in, and re-adding x_out.
        # Or, just check creates_4ap_with_x(S - {x_out}, x_in).
        # To avoid set copy overhead, we can modify the check function or just do it.
        # Given n=512, set copy is cheap.
        S_prime = S - {x_out}
        return not creates_4ap_with_x(S_prime, x_in)

    # Initial Greedy Construction
    current_set = set()
    for i in range(1, n + 1):
        if not creates_4ap_with_x(current_set, i):
            current_set.add(i)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    import time
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    all_elements = list(range(1, n + 1))
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        
        # Phase 1: Try to add elements
        added = False
        # Shuffle candidates to avoid bias
        candidates = [x for x in all_elements if x not in current_set]
        random.shuffle(candidates)
        for x in candidates:
            if not creates_4ap_with_x(current_set, x):
                current_set.add(x)
                added = True
        
        if added:
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
                continue # Restart local search phase
        
        # Phase 2: Try Swaps
        # If we couldn't add, try to swap elements to change the structure.
        # We try a limited number of swaps to avoid getting stuck in a long swap loop.
        swapped = False
        # To make this efficient, we can try random swaps first, then systematic if needed.
        # Systematic swap check is O(|S| * (n-|S|)) which is ~ 250*250 = 62500 checks.
        # Each check is O(n) ~ 500 ops. Total ~ 30M ops. Might be too slow per iteration if done repeatedly.
        # Let's try a subset of swaps or random swaps.
        
        # Try up to 100 random swaps
        for _ in range(100):
            if not current_set: break
            x_out = random.choice(list(current_set))
            x_in = random.choice(candidates) # candidates is non-empty if we are stuck and size < n
            if x_in in current_set: continue
            if valid_swap(current_set, x_out, x_in):
                current_set.remove(x_out)
                current_set.add(x_in)
                swapped = True
                # After a swap, try to add again?
                # For simplicity, break and go back to Phase 1.
                break
        
        if swapped:
            continue # Go back to Phase 1
        
        # Phase 3: Perturbation (Ruin and Recreate)
        # Remove k random elements
        k = random.randint(2, 5)
        if len(current_set) > k:
            to_remove = random.sample(list(current_set), k)
            for x in to_remove:
                current_set.remove(x)
            
            # Greedy Refill
            refill_order = list(all_elements)
            random.shuffle(refill_order)
            for x in refill_order:
                if x not in current_set:
                    if not creates_4ap_with_x(current_set, x):
                        current_set.add(x)
            
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)

    return sorted(best_set)
