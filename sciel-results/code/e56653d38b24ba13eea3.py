"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 88  # safety margin before 90s kill
    
    # Compute LCM of 1..22 for exact integer arithmetic
    L = reduce(lcm, range(1, 23))
    
    # Precompute the integer values: v[a] = L // a
    vals = {a: L // a for a in range(1, 23)}
    elements = list(range(1, 23))
    
    # Sort elements by their value in descending order.
    # Larger values tend to be more "independent" and help in the superincreasing check.
    elements_sorted = sorted(elements, key=lambda x: vals[x], reverse=True)
    
    def is_superincreasing(subset_vals):
        """
        Check if the sorted subset values form a superincreasing sequence.
        If v_i > sum(v_j for j < i), then all subset sums are distinct.
        """
        if not subset_vals:
            return True
        sorted_vals = sorted(subset_vals)
        current_sum = 0
        for v in sorted_vals:
            if v <= current_sum:
                return False
            current_sum += v
        return True

    def is_dissociated_fast(subset):
        """
        Check if all subset sums of vals[a] for a in subset are distinct.
        Uses superincreasing check first as a fast path.
        """
        k = len(subset)
        if k == 0:
            return True
        
        v_list = [vals[a] for a in subset]
        
        # Fast path: Superincreasing
        if is_superincreasing(v_list):
            return True
            
        # Slow path: Generate all subset sums and check for duplicates
        sums = {0}
        for v in v_list:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    # We want to find the maximum cardinality subset.
    # Try from largest possible size (22) down to 1.
    best_result = [1]  # fallback
    
    for k in range(22, 0, -1):
        if time.time() > deadline - 2:
            break
        
        found = False
        
        # Backtracking search for a valid subset of size k
        # We use the sorted elements (by value descending) to guide the search
        
        def backtrack(index, chosen, current_sums):
            """
            index: current index in elements_sorted
            chosen: list of chosen elements
            current_sums: set of subset sums of chosen elements
            """
            if time.time() > deadline - 1:
                return None
            
            needed = k - len(chosen)
            if needed == 0:
                return chosen[:]
            
            # Not enough elements left
            if index + needed > len(elements_sorted):
                return None
            
            # Try to pick elements from index to end
            for i in range(index, len(elements_sorted)):
                elem = elements_sorted[i]
                v = vals[elem]
                
                # Check if adding v creates a collision
                # New sums: s + v for s in current_sums
                # Must be disjoint from current_sums
                # Optimization: if current_sums is small, iterate.
                # If current_sums is large, this is expensive.
                
                valid = True
                # We don't need to build the full new_sums set if we just want to check validity
                # But we need it for recursion if valid.
                # Let's check validity first.
                for s in current_sums:
                    if (s + v) in current_sums:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # If valid, construct new_sums
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + v)
                
                # Recurse
                result = backtrack(i + 1, chosen + [elem], new_sums)
                if result is not None:
                    return result
            
            return None

        # Start search
        # To speed up, we can try to satisfy the superincreasing condition first?
        # Or just use the standard backtracking which handles all cases.
        
        # Heuristic: If k is large, finding a superincreasing subset might be easier?
        # The set of values is fixed.
        
        result = backtrack(0, [], {0})
        
        if result is not None:
            best_result = sorted(result)
            found = True
            break  # Found maximum cardinality, we're done
            
        # If not found, try smaller k
    
    return best_result
# EVOLVE-BLOCK-END
