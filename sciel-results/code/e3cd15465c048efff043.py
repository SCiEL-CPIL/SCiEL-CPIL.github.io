import time
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Precompute weights
    # L = lcm(1, ..., 22)
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i
        
    # Helper: check if a set of indices has all distinct subset sums of weights
    # Returns the set of subset sums if valid, else None
    def get_subset_sums(indices):
        if not indices:
            return {0}
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums:
                    return None
                new_sums.add(ns)
            sums.update(new_sums)
        return sums

    # Faster incremental check
    def check_addable(current_sums, element_idx):
        w = weights[element_idx]
        new_sums = set()
        for s in current_sums:
            ns = s + w
            if ns in current_sums:
                return None
            new_sums.add(ns)
        return new_sums

    all_elements = list(range(1, 23))
    best_set = []
    best_sums = None
    
    # Multiple greedy strategies
    # Strategy 1: Largest weights first (smallest indices) -> Most restrictive?
    # Strategy 2: Smallest weights first (largest indices) -> Least restrictive?
    # Strategy 3: Alternating
    
    orders = [
        all_elements[::-1],  # 22 down to 1 (smallest weights first)
        all_elements,        # 1 up to 22 (largest weights first)
        # Interleaved
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12]
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        current_sums = {0}
        
        for e in order:
            if time.time() > deadline:
                break
            new_sums = check_addable(current_sums, e)
            if new_sums is not None:
                current.append(e)
                current_sums.update(new_sums)
        
        if len(current) > len(best_set):
            best_set = current[:]
            best_sums = set(current_sums)
            
    # Local search: try to add remaining elements
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable(best_sums, e)
                if new_sums is not None:
                    best_set.append(e)
                    best_sums.update(new_sums)
                    remaining.remove(e)
                    improved = True
                    
    # Local search: try swaps
    # If we are stuck, try removing one element and adding two others
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        # We try to improve the size by 1 or more
        # For each element in best_set, remove it, then try to greedily add from remaining
        # If we can add > 1 element, we have improved the size
        
        # To avoid expensive full rebuilds, we can try to just check if removing `rem` allows adding `add1` and `add2`
        # But rebuilding the sum set after removal is $O(2^k)$.
        # With integer sums, it's fast enough for k <= 12 or 13.
        
        best_size = len(best_set)
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            test_set = best_set[:i] + best_set[i+1:]
            
            # Rebuild sums for test_set
            test_sums = get_subset_sums(test_set)
            if test_sums is None:
                continue # Should not happen if best_set was valid
                
            # Try to add elements from remaining
            added_count = 0
            new_remaining = remaining[:]
            # Sort remaining by some heuristic? Maybe just iterate.
            for e in new_remaining:
                if time.time() > deadline:
                    break
                new_sums = check_addable(test_sums, e)
                if new_sums is not None:
                    test_set.append(e)
                    test_sums.update(new_sums)
                    added_count += 1
                    # We want to see if we can add MORE than we removed (1)
                    # So if added_count >= 1, size is same. If >= 2, size improves.
                    # Let's just greedily add as many as possible to this candidate
                    # and check if final size > best_size
            
            if len(test_set) > best_size:
                best_set = test_set
                best_sums = test_sums
                best_size = len(best_set)
                remaining = [e for e in all_elements if e not in best_set]
                # Restart the "try to add remaining" loop? 
                # Or just break and let the next iteration of the outer loop handle it?
                # Since we improved, we should probably try to add more from the new remaining set.
                # But the loop structure is a bit rigid. 
                # Let's just update best_set and break out of the swap loop to re-run the "add remaining" phase?
                # Actually, the "add remaining" phase is already done. 
                # If we found a better set, we should try to add more to IT.
                # Let's set a flag to re-run the add phase.
                improved = True 
                break # Break inner loop, re-run add phase? 
                # The code structure doesn't easily support re-running. 
                # Let's just update and continue. The next iteration of the outer `for i` loop might find more, 
                # but we should definitely try to add to the new best_set.
                
    # Final "add remaining" pass on the potentially improved best_set
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable(best_sums, e)
                if new_sums is not None:
                    best_set.append(e)
                    best_sums.update(new_sums)
                    remaining.remove(e)
                    improved = True

    # Fallback
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
