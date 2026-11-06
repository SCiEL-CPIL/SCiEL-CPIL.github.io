# EVOLVE-BLOCK-START
import time

def _oem(indices):
    """
    Generate Odd-Even Merge network for a list of wire indices.
    Returns a list of comparators.
    """
    n = len(indices)
    if n <= 1:
        return []
    if n == 2:
        return [(indices[0], indices[1])]
    
    # Split into even and odd positions in the list
    even_indices = indices[0::2]
    odd_indices = indices[1::2]
    
    # Recursively sort even and odd
    comps = []
    comps.extend(_oem(even_indices))
    comps.extend(_oem(odd_indices))
    
    # Merge step
    # Compare adjacent pairs in the merged sequence
    # The merged sequence is interleaved:
    # e0, o0, e1, o1, ...
    # We compare (e0, o0), (e1, o1), ...
    # Then recursively merge the even and odd positions of the RESULT?
    # No, the standard Odd-Even Merge merge step is:
    # 1. Compare (0,1), (2,3), ...
    # 2. Recursively merge the even positions of the result.
    # 3. Recursively merge the odd positions of the result.
    
    # Wait, the indices in the "result" are the same wires.
    # The "even positions of the result" are indices[0], indices[2], ...
    # Which are exactly `even_indices`?
    # No, after the first compare, the values have moved.
    # But the wires are fixed.
    # The "even positions" refer to the logical positions in the sorted output.
    # In Odd-Even Merge, the recursion is on the *positions*, not the values.
    
    # The merge step for Odd-Even Merge:
    # 1. Compare (indices[0], indices[1]), (indices[2], indices[3]), ...
    # 2. Recursively merge indices[0::2] (even positions)
    # 3. Recursively merge indices[1::2] (odd positions)
    
    for i in range(0, n, 2):
        comps.append((indices[i], indices[i+1]))
    
    # Recursively merge even positions
    comps.extend(_oem(indices[0::2]))
    # Recursively merge odd positions
    comps.extend(_oem(indices[1::2]))
    
    return comps

def construct():
    # Use Odd-Even Merge for 18 wires
    indices = list(range(N))
    return _oem(indices)
# EVOLVE-BLOCK-END
