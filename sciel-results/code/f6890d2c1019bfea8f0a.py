"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    start_time = time.time()
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def check_add(element, elements, int_vals, subset_sums, L):
        """Check if adding 'element' to the set keeps all subset sums distinct.
        Returns (bool, new_subset_sums) if we need to update."""
        v = L // element
        # Check if v + s collides with any existing sum s'
        # i.e., does there exist s in subset_sums such that v + s is also in subset_sums?
        # Or more precisely: the new sums are {s + v for s in subset_sums}
        # We need all 2^(k+1) sums to be distinct, meaning no two of the original 2^k
        # and the new 2^k overlap, AND the new ones are all distinct among themselves.
        # New ones are distinct among themselves iff original ones are (which they are).
        # So we just need: (s + v) not in subset_sums for all s in subset_sums
        for s in subset_sums:
            if (s + v) in subset_sums:
                return False, subset_sums
        return True, subset_sums
    
    def build_subset_sums(int_vals):
        """Build all subset sums from integer values."""
        sums = {0}
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return None  # collision detected
                new_sums.add(ns)
            sums.update(new_sums)
        return sums
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
        k = len(elements)
        if k == 0:
            return True
        if k > 20:
            return False  # too expensive, assume not dissociated for pruning
        
        L = reduce(lcm, elements)
        int_vals = [L // x for x in elements]
        sums = build_subset_sums(int_vals)
        return sums is not None
    
    # Greedy construction with incremental subset sum tracking
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Multiple diverse orderings
    orderings = [
        sorted(candidates, reverse=True),  # Largest first
        sorted(candidates),                 # Smallest first
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19],
        [16, 12, 8, 4, 2, 20, 15, 10, 5, 22, 11, 6, 3, 18, 9, 7, 14, 21, 13, 19, 17, 1],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
        # Additional strategic orderings
        [22, 21, 19, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 20, 18],
        [18, 22, 16, 20, 14, 12, 10, 8, 6, 4, 2, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        current = []
        current_L = 1
        current_int_vals = []
        current_sums = {0}
        
        for elem in order:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            
            # Compute new LCM
            new_L = lcm(current_L, elem) if current_L > 0 else elem
            if current_L == 1:
                new_L = elem
            
            # Scale existing int_vals to new L
            if new_L != current_L:
                scale = new_L // current_L if current_L > 0 else 1
                current_int_vals = [v * scale for v in current_int_vals]
                current_sums = {s * scale for s in current_sums}
            
            current_L = new_L
            v = new_L // elem
            
            # Check if adding v creates a collision
            collision = False
            for s in current_sums:
                if (s + v) in current_sums:
                    collision = True
                    break
            
            if not collision:
                current.append(elem)
                current_int_vals.append(v)
                # Update subset sums
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + v)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local search: try to add elements
    if best_set and time.time() < deadline - 10:
        remaining = [x for x in candidates if x not in best_set]
        
        # Try adding elements in various orders
        for order in [sorted(remaining, reverse=True), sorted(remaining), sorted(remaining, key=lambda x: -x)]:
            if time.time() > deadline - 10:
                break
            for elem in order:
                if time.time() > deadline - 10:
                    break
                if elem in best_set:
                    continue
                test_set = best_set + [elem]
                if check_dissociated(test_set):
                    best_set = test_set
                    best_size = len(best_set)
    
    # Swap-based local search
    if time.time() < deadline - 15:
        for _ in range(5):  # Multiple rounds
            if time.time() > deadline - 15:
                break
            progress = False
            for i in range(len(best_set)):
                if time.time() > deadline - 15:
                    break
                for new_elem in candidates:
                    if time.time() > deadline - 15:
                        break
                    if new_elem in best_set:
                        continue
                    candidate_set = best_set[:i] + [new_elem] + best_set[i+1:]
                    candidate_set.sort()
                    if check_dissociated(candidate_set):
                        best_set = candidate_set
                        progress = True
                        # Try to add more after swap
                        for add_elem in candidates:
                            if time.time() > deadline - 15:
                                break
                            if add_elem in best_set:
                                continue
                            test_set = best_set + [add_elem]
                            if check_dissociated(test_set):
                                best_set = test_set
                                best_size = len(best_set)
                        break
                if progress:
                    break
            if not progress:
                break
    
    # Final polish
    if time.time() < deadline - 5:
        for elem in sorted(candidates, reverse=True):
            if time.time() > deadline - 5:
                break
            if elem in best_set:
                continue
            test_set = best_set + [elem]
            if check_dissociated(test_set):
                best_set = test_set
                best_size = len(best_set)
    
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
