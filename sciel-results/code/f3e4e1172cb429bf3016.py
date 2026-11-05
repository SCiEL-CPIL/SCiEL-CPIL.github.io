import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 80  # Safety margin
    
    # Compute LCM of 1..22
    lcm_val = 1
    for i in range(1, 23):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    
    # Integer coefficients: c[i] = lcm_val // (i+1) for element i+1
    # We work with the set of integers C = {L/i for i in 1..22}
    # A subset S of [1..22] is dissociated iff {L/s for s in S} has distinct subset sums.
    c = [lcm_val // (i+1) for i in range(22)]
    
    # Sort the coefficients. 
    # Heuristic: Larger coefficients (smaller reciprocals) might be less prone to collisions 
    # with smaller sums? Or maybe smaller first?
    # Let's sort by value. 
    # Note: The value of the element in the original set [1..22] is i+1.
    # We need to return the original values.
    # Let's keep track of (coefficient, original_value)
    items = sorted([(c[i], i+1) for i in range(22)], key=lambda x: x[0])
    
    best_set = []
    
    # Greedy initialization to have a fallback
    current_sums = {0}
    current_set = []
    for coef, val in items:
        # Check if adding coef causes collision
        # New sums: current_sums U (current_sums + coef)
        # Collision if (current_sums + coef) intersects current_sums
        # i.e., exists s1, s2 in current_sums such that s1 + coef = s2
        # i.e., coef = s2 - s1
        valid = True
        for s in current_sums:
            if (s + coef) in current_sums:
                valid = False
                break
        if valid:
            # Add to sums
            new_sums = current_sums | {s + coef for s in current_sums}
            current_sums = new_sums
            current_set.append(val)
            
    best_set = current_set[:]
    
    # Try to find larger sets using Branch and Bound
    # We search for k from 22 down to len(best_set)+1
    # If we find one, we return it.
    
    # To speed up B&B, we can pre-filter items that are "too large" or "too small"?
    # No, just use the sorted list.
    
    def search(k, idx, current, sums):
        """
        Recursively build a set of size k.
        idx: index in 'items' to start considering.
        current: list of original values chosen.
        sums: set of current subset sums.
        Returns True if a valid set of size k is found.
        """
        nonlocal best_set
        if time.time() > deadline - 2:
            return False
        
        if len(current) == k:
            best_set = current[:]
            return True
        
        # Pruning: not enough items left
        if len(current) + (22 - idx) < k:
            return False
            
        # If we have more than k, we don't add more
        # We need to pick exactly k - len(current) more items from items[idx:]
        
        # Optimization: If we are close to the deadline, relax to just finding *any* improvement?
        # No, the contract says maximize. But if we can't prove max, return best.
        
        for i in range(idx, 22):
            if time.time() > deadline - 2:
                return False
                
            coef, val = items[i]
            
            # Check collision
            # If any (s + coef) is in sums, collision.
            # This check is O(|sums|).
            # |sums| is 2^len(current).
            valid = True
            for s in sums:
                if (s + coef) in sums:
                    valid = False
                    break
            
            if valid:
                new_sums = sums | {s + coef for s in sums}
                if search(k, i + 1, current + [val], new_sums):
                    return True
                    
        return False

    # Try to find larger sets
    # Start from 22 down to len(best_set) + 1
    for k in range(22, len(best_set), -1):
        if time.time() > deadline - 5:
            break
        
        # Reset search
        # We can use a quick greedy check to see if k is even possible?
        # No, just run B&B.
        
        # To make B&B faster, we can order items differently for each k?
        # Or just use the pre-sorted list.
        
        # One optimization: If we fail for k, we might fail for k+1?
        # Not necessarily, but if a set of size k+1 exists, a subset of size k exists.
        # So if we find a set of size k, we know k is possible.
        # We want the MAXIMUM k.
        
        # We should search for the largest k first.
        # If search(22) fails, try 21, etc.
        
        if search(k, 0, [], {0}):
            # Found a set of size k
            # Since we are iterating downwards, this is the maximum.
            break

    # Final verification
    if best_set:
        # Verify the returned set is actually valid
        # Just in case of bugs in the logic
        final_sums = {0}
        valid = True
        for val in best_set:
            coef = lcm_val // val
            new_sums = set()
            for s in final_sums:
                ns = s + coef
                if ns in final_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                break
            final_sums = final_sums | new_sums
        
        if not valid:
            # Fallback to the greedy result which was verified in construction
            # Re-run greedy to get a clean state
            current_sums = {0}
            current_set = []
            for coef, val in items:
                valid = True
                for s in current_sums:
                    if (s + coef) in current_sums:
                        valid = False
                        break
                if valid:
                    new_sums = current_sums | {s + coef for s in current_sums}
                    current_sums = new_sums
                    current_set.append(val)
            best_set = current_set
    
    return best_set

# Helper for local testing
if __name__ == "__main__":
    res = construct()
    print(f"Found set: {res}")
    print(f"Size: {len(res)}")
