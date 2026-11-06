import time
import random

def construct():
    deadline = time.time() + 80
    n = 12
    
    def subset_sums_set(arr):
        """Compute all subset sums as a set."""
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            sums |= new_sums
        return sums

    def is_valid(arr):
        """Check if all subset sums are distinct. Returns True if valid."""
        if len(set(arr)) != len(arr):
            return False
        sums = subset_sums_set(arr)
        return len(sums) == (1 << len(arr))

    def get_min_valid_val(arr, idx, max_val):
        """Find the smallest value < max_val that can replace arr[idx] while keeping validity."""
        current = arr[idx]
        others = arr[:idx] + arr[idx+1:]
        used_vals = set(others)
        
        # We need to check if replacing arr[idx] with val keeps all subset sums distinct.
        # Subset sums of 'others' are fixed. Let S_others be the set of subset sums of others.
        # If we add val, the new sums are S_others union (S_others + val).
        # These must be disjoint (except for the implicit 0 if val=0, but val>0).
        # Actually, S_others and (S_others + val) must be disjoint.
        # This is equivalent to: no s1, s2 in S_others such that s1 - s2 = val.
        # Or simply: check intersection.
        
        s_others = subset_sums_set(others)
        
        # Optimization: sort s_others? No, set intersection is O(min(n,m)) on average.
        # But we can precompute the set of differences? 
        # For each candidate val, we check if (s_others + val) & s_others is empty.
        
        # To speed up, we can precompute the set of all differences s1 - s2 for s1, s2 in s_others?
        # That's O(|S|^2) which is 2^11 * 2^11 = 4M. Might be too slow if done repeatedly.
        
        # Let's just iterate.
        for val in range(1, max_val):
            if val in used_vals:
                continue
            # Check distinctness
            shifted = {s + val for s in s_others}
            if s_others & shifted:
                continue
            # Also need to ensure val is distinct from others (checked via used_vals)
            # And positive (checked via range)
            return val
        return None

    # Start with powers of 2
    arr = [1 << i for i in range(n)]
    
    # Greedy reduction pass 1: try to reduce each element from smallest to largest
    # This is a known heuristic for "minimal superincreasing" like sequences
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for idx in range(n):
            if time.time() > deadline - 5:
                break
            current = arr[idx]
            if current <= 1:
                continue
            new_val = get_min_valid_val(arr, idx, current)
            if new_val is not None and new_val < current:
                arr[idx] = new_val
                improved = True

    # If we have time, try randomized local search to break local optima
    # The greedy pass might get stuck if reducing a small element prevents reducing a large one later,
    # or if the order matters.
    
    best_arr = list(arr)
    best_max = max(arr)
    
    if time.time() < deadline - 10:
        # Randomized restarts or perturbations
        # Strategy: Pick a random index, try to swap or adjust.
        # Since the problem is small, we can try random valid mutations.
        
        # A better random search: 
        # 1. Keep the greedy result.
        # 2. Try to randomly perturb one element to a smaller value, even if it breaks validity temporarily? No, must stay valid.
        # 3. Try to find a different valid sequence.
        
        # Let's try a simple hill climbing with random restarts from the greedy solution.
        # We will try to reduce the max element specifically.
        
        # Actually, the greedy pass from smallest to largest is quite strong for this specific "distinct subset sums" problem 
        # because it effectively constructs a "minimal" sequence in lexicographical order of the sorted array?
        # Not exactly, but close.
        
        # Let's try a different order: reduce from largest to smallest?
        # Or try multiple orders.
        
        orders = [
            list(range(n)),
            list(range(n-1, -1, -1)),
            list(range(n-1, -1, -1, 2)), # even indices first
            list(range(n-1, -1, -1, 2)) + list(range(n-2, -1, -1, 2)) # odd then even
        ]
        
        for order in orders:
            if time.time() > deadline - 5:
                break
            # Reset to powers of 2? Or start from current best?
            # Starting from powers of 2 with a different order might find a different local minimum.
            # Let's start from powers of 2 again for each order to see if we get a better max.
            arr_temp = [1 << i for i in range(n)]
            
            improved = True
            while improved:
                if time.time() > deadline - 5:
                    break
                improved = False
                for idx in order:
                    if time.time() > deadline - 5:
                        break
                    current = arr_temp[idx]
                    if current <= 1:
                        continue
                    new_val = get_min_valid_val(arr_temp, idx, current)
                    if new_val is not None and new_val < current:
                        arr_temp[idx] = new_val
                        improved = True
            
            if max(arr_temp) < best_max:
                best_max = max(arr_temp)
                best_arr = list(arr_temp)
                
    # Final check
    if not is_valid(best_arr):
        return [1 << i for i in range(n)]
        
    return best_arr
