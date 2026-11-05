"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    deadline = time.time() + 55  # Safety margin before 90s limit
    
    # LCM of 1..22
    D = 232792560
    vals = {x: D // x for x in range(1, 23)}
    
    # We want to find the largest subset.
    # Based on inspirations, size 16 is likely achievable (score 1.77).
    # Let's try to find a set of size 16, then 15, etc.
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19] # Fallback size 9
    best_size = 9
    
    all_elements = list(range(1, 23))
    
    # Heuristic: Smaller denominators (larger values) are more "expensive"
    # and likely to cause collisions. 
    # Let's sort candidates by value descending (denominator ascending) for the search order.
    # This tends to fill the "large" sums first.
    
    def is_valid_addition(current_sums, v):
        """
        Check if adding value v to the set of sums creates a collision.
        current_sums is a set of integers.
        """
        # We need to check if (s + v) is in current_sums for any s in current_sums.
        # This is equivalent to: (current_sums + v) & current_sums != empty.
        # To avoid creating a new set every time, we can iterate.
        # Optimization: If we store sums in a sorted list or use bitsets, it's faster.
        # But for k<=16, set of size <= 65536 is fine.
        
        # A faster check: 
        # If we have the set of sums S, adding v creates new sums S' = S U (S+v).
        # Collision exists if (S+v) intersects S.
        # This is the same as: does S contain any difference s1 - s2 = v?
        # Or simpler: just check the intersection.
        
        # Iterating over the smaller set is good. Here both are size 2^m.
        # Let's just do the intersection check.
        # Note: v is positive.
        
        # We can optimize by noting that max(sum) + v is the new max.
        # We only need to check s such that s + v <= max(current_sums).
        # But simple iteration is likely fast enough for 65k elements.
        
        for s in current_sums:
            if s + v in current_sums:
                return False
        return True

    def backtrack(start_idx, remaining, current_elements, current_sums, elements_order):
        """
        Recursive backtracking to find a valid set of size 'target_size'.
        elements_order: list of elements to consider.
        """
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
            
        if remaining == 0:
            # Found a valid set of the target size
            best_set = list(current_elements)
            best_size = len(current_elements)
            return True
        
        # Pruning: not enough elements left
        if len(elements_order) - start_idx < remaining:
            return False
            
        # Iterate through available elements
        for i in range(start_idx, len(elements_order)):
            if time.time() > deadline:
                return False
                
            x = elements_order[i]
            v = vals[x]
            
            # Check validity
            if is_valid_addition(current_sums, v):
                # Add x
                current_elements.append(x)
                
                # Update sums
                old_sums = current_sums
                new_sums = set(old_sums)
                # Optimization: Only add s+v if it's not already in old_sums?
                # Actually, we just need the new set for the next recursion.
                for s in old_sums:
                    new_sums.add(s + v)
                current_sums = new_sums
                
                if backtrack(i + 1, remaining - 1, current_elements, current_sums, elements_order):
                    return True
                
                # Undo
                current_elements.pop()
                current_sums = old_sums
                
        return False

    # Try to find sets of decreasing size
    # We start with a high target to maximize score.
    # If we find size 16, we stop.
    
    for target_size in range(16, 8, -1):
        if target_size <= best_size:
            break
            
        # Try multiple random orders to escape local minima / bad search paths
        # We use a mix of deterministic and random orders.
        
        # Order 1: Denominators ascending (Values descending)
        # This is often good because large values are harder to fit.
        order1 = sorted(all_elements, key=lambda x: -vals[x])
        
        # Order 2: Denominators descending (Values ascending)
        order2 = sorted(all_elements, key=lambda x: vals[x])
        
        # Order 3: Random shuffles
        orders = [order1, order2]
        for _ in range(50):
            o = all_elements[:]
            random.shuffle(o)
            orders.append(o)
            
        found = False
        for order in orders:
            if time.time() > deadline:
                break
            if found:
                break
                
            current_elements = []
            current_sums = {0}
            
            # Use a fixed random seed for reproducibility if needed, but random is fine here
            if backtrack(0, target_size, current_elements, current_sums, order):
                found = True
                break
                
        if found:
            break
            
    return best_set
# EVOLVE-BLOCK-END
