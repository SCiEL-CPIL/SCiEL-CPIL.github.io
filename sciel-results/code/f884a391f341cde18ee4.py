import time
import math
import random
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    # Precompute weights: L = lcm(1, ..., 22)
    # This allows us to use integer arithmetic for subset sum checks.
    # If we select indices S, we check if {L/s : s in S} is dissociated.
    # This is equivalent to {1/s : s in S} being dissociated because
    # a linear dependence sum c_i / s_i = 0 with c_i in {-1,0,1}
    # is equivalent to sum c_i * (L/s_i) = 0.
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    # weights[i] is the integer weight corresponding to 1/i
    # We work with these integers to avoid floating point errors and speed up set operations.
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i

    all_elements = list(range(1, 23))
    
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

    best_set = []
    best_sums = None
    best_size = 0
    
    # Strategy 1: Multiple deterministic greedy orders
    # The order in which we add elements matters significantly.
    # "Smallest weights first" (largest indices) tends to allow more elements because
    # they create smaller gaps in the sum space, but actually, it's the opposite:
    # Large weights create large gaps, making it easier to fit more small weights?
    # Let's test several orders.
    
    orders = [
        all_elements[::-1],  # 22 down to 1 (smallest weights first)
        all_elements,        # 1 up to 22 (largest weights first)
        # Interleaved
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
        # Randomized shuffles (deterministic seed for reproducibility)
        [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21],
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
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
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            best_sums = set(current_sums)
            
    # Strategy 2: Local Search / Hill Climbing
    # Try to improve the best set found so far by swapping elements.
    # If we have a set S of size k, and we remove one element x, we can try to add
    # multiple elements from the remaining pool. If we can add > 1, we increase size.
    
    if best_set and time.time() < deadline:
        # Ensure best_set is sorted for consistent comparison
        best_set.sort()
        
        # Re-verify best_set just in case
        # (It should be valid by construction)
        
        # We will perform several rounds of local search
        # In each round, we try to improve the set.
        
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            current_set = best_set[:]
            current_sums = best_sums
            current_size = len(current_set)
            remaining = [e for e in all_elements if e not in current_set]
            
            improved = False
            
            # Try removing each element from current_set
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                
                removed = current_set[i]
                # Rebuild sums without 'removed'
                # This is O(2^k * k) if we do it naively, but we can be smarter.
                # However, k is small (<= 12 or so), so rebuilding is fast.
                
                test_set = current_set[:i] + current_set[i+1:]
                test_sums = get_subset_sums(test_set)
                if test_sums is None:
                    continue # Should not happen
                
                # Try to add elements from 'remaining' to test_set
                # Greedily add as many as possible
                added_count = 0
                new_test_sums = test_sums
                new_test_set = test_set[:]
                
                # Sort remaining by some heuristic? 
                # Maybe try all of them in order.
                for e in remaining:
                    if time.time() > deadline:
                        break
                    new_sums = check_addable(new_test_sums, e)
                    if new_sums is not None:
                        new_test_set.append(e)
                        new_test_sums.update(new_sums)
                        added_count += 1
                
                # If we added more elements than we removed (1), we improved size
                if added_count > 1:
                    new_size = len(new_test_set)
                    if new_size > current_size:
                        current_set = new_test_set
                        current_sums = new_test_sums
                        current_size = new_size
                        remaining = [e for e in all_elements if e not in current_set]
                        improved = True
                        # We found an improvement, restart the local search loop
                        # to try further improvements from this new state
                        break # Break the 'for i' loop
            
            if improved:
                # Update best
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                    best_sums = current_sums
                    # Continue local search from this new best
                    continue
            else:
                # No improvement found in this round, try a different perturbation?
                # Or stop if we are stuck.
                # Let's try a random restart or different swap strategy.
                # For now, if no improvement, we might be at a local optimum.
                # We can try swapping 2 elements for 3, etc., but that's complex.
                # Let's try a few random shuffles of the greedy order to see if we find a better global start.
                
                # If we haven't improved, break to save time? 
                # Or try a few more random greedy starts.
                if _ > 0: 
                    # We are in the local search phase, if no improvement, we might be stuck.
                    # Let's break to avoid wasting time on the same local optimum.
                    break
                else:
                    # First iteration of local search, if no improvement, try random greedy
                    pass

    # Strategy 3: Randomized Greedy with Restart
    # If local search didn't find a huge improvement, try randomizing the greedy order.
    
    if time.time() < deadline:
        # Try 50 random shuffles
        for _ in range(50):
            if time.time() > deadline:
                break
            
            random.shuffle(all_elements)
            current = []
            current_sums = {0}
            
            for e in all_elements:
                if time.time() > deadline:
                    break
                new_sums = check_addable(current_sums, e)
                if new_sums is not None:
                    current.append(e)
                    current_sums.update(new_sums)
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
                best_sums = set(current_sums)
                
                # If we improved, do a quick local search on this new best
                if time.time() < deadline:
                    # Quick local search: try to add remaining
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
