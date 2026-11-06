import time
from math import lcm
from functools import reduce

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    # Deadline guard: 85 seconds to be safe against 90s kill
    deadline = time.time() + 85
    
    # Compute LCM of 1..22 to convert reciprocal sums to integer sums
    L = reduce(lcm, range(1, 23))
    
    # Map each element a to its integer contribution L // a
    vals = {a: L // a for a in range(1, 23)}
    
    # Elements sorted by value descending.
    # Larger values are more "constraining" as they create larger sums
    # and are more likely to cause collisions.
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    # Precompute values in this sorted order for faster access
    sorted_vals = [vals[a] for a in elements]
    
    def is_dissociated(current_sums, v):
        """
        Check if adding value v to the set of current sums results in
        a set with distinct sums.
        current_sums: set of integers
        v: integer
        Returns True if (current_sums + v) is disjoint from current_sums.
        """
        # Check if any s + v is in current_sums
        # Equivalent to: (current_sums + v) & current_sums == empty set
        for s in current_sums:
            if s + v in current_sums:
                return False
        return True

    def find_subset_of_size(target_size):
        """
        Use backtracking to find a subset of exactly target_size elements
        from the sorted elements list that is dissociated.
        Returns the subset (list of original element values) or None.
        """
        current_sums = {0}
        current_subset = []
        
        # We need to pick target_size elements from the list of 22.
        # We iterate through the sorted elements.
        
        def backtrack(idx, current_size, current_sums, current_subset):
            if time.time() > deadline:
                return None
            
            # Pruning 1: Not enough elements left to reach target_size
            if current_size + (len(elements) - idx) < target_size:
                return None
            
            # Base case: We found a valid subset of the required size
            if current_size == target_size:
                return list(current_subset)
            
            # Pruning 2: If we have picked enough, we don't need to pick more
            # (This is handled by the base case, but we can also stop expanding
            #  if we just need to verify existence, but we need the subset to return)
            
            # Try to include elements from idx to end
            for i in range(idx, len(elements)):
                if time.time() > deadline:
                    return None
                
                a = elements[i]
                v = sorted_vals[i]
                
                # Check if adding v is valid
                if is_dissociated(current_sums, v):
                    # Calculate new sums
                    # new_sums = current_sums | {s + v for s in current_sums}
                    # We can optimize by checking collision first (done above)
                    # then generating the set
                    new_sums = current_sums | {s + v for s in current_sums}
                    
                    # Recurse
                    # We pick element at index i, so next index is i + 1
                    current_subset.append(a)
                    result = backtrack(i + 1, current_size + 1, new_sums, current_subset)
                    current_subset.pop()
                    
                    if result is not None:
                        return result
            
            return None

        return backtrack(0, 0, {0}, [])

    # Search for the maximum cardinality
    # We try sizes from 22 down to 1.
    # As soon as we find a valid subset of size T, T is the maximum.
    for size in range(22, 0, -1):
        if time.time() > deadline - 1:
            break
            
        result = find_subset_of_size(size)
        if result is not None:
            return result
        
        # If we didn't find one, continue to smaller sizes
    
    # Fallback: return a single element (always valid)
    return [1]
