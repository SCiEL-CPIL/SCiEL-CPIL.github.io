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
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    # Precompute values in this sorted order for faster access
    sorted_vals = [vals[a] for a in elements]
    
    def is_dissociated_with_diffs(diffs, v):
        """
        Check if adding value v is valid given the current set of forbidden differences.
        Adding v is invalid iff v is in diffs (since then some s1 + v = s2, i.e., v = s2 - s1).
        """
        return v not in diffs

    def add_to_diffs(diffs, v):
        """
        Update the difference set after adding v to the set of sums.
        new_diffs = diffs ∪ {d + v : d in diffs} ∪ {d - v : d in diffs}
        """
        new_diffs = set(diffs)
        for d in diffs:
            new_diffs.add(d + v)
            new_diffs.add(d - v)
        return new_diffs

    def find_subset_of_size(target_size):
        """
        Use backtracking to find a subset of exactly target_size elements
        from the sorted elements list that is dissociated.
        Returns the subset (list of original element values) or None.
        """
        
        def backtrack(idx, current_size, diffs, current_subset):
            if time.time() > deadline:
                return None
            
            # Pruning 1: Not enough elements left to reach target_size
            if current_size + (len(elements) - idx) < target_size:
                return None
            
            # Base case: We found a valid subset of the required size
            if current_size == target_size:
                return list(current_subset)
            
            # Try to include elements from idx to end
            for i in range(idx, len(elements)):
                if time.time() > deadline:
                    return None
                
                a = elements[i]
                v = sorted_vals[i]
                
                # Check if adding v is valid using the difference set
                if is_dissociated_with_diffs(diffs, v):
                    # Calculate new diffs
                    new_diffs = add_to_diffs(diffs, v)
                    
                    # Recurse
                    current_subset.append(a)
                    result = backtrack(i + 1, current_size + 1, new_diffs, current_subset)
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
