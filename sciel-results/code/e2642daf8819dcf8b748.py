import time
import itertools

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    P = 101
    D = 10
    elements = list(range(1, D + 1))
    
    # Precompute modular inverses
    inv = [0] * (D + 1)
    for i in range(1, D + 1):
        inv[i] = pow(i, -1, P)
        
    # Generate all non-empty subsets with their sums and sizes
    # We store them in a list: (size, sum, tuple_of_elements)
    all_subsets = []
    
    # Efficiently generate subsets and their sums
    # Using bit masks is fast enough for D=10
    for mask in range(1, 1 << D):
        res = 0
        elems = []
        # Calculate sum and elements
        # We can also count bits for size
        for i in range(D):
            if mask & (1 << i):
                x = i + 1
                res = (res + inv[x]) % P
                elems.append(x)
        sz = len(elems)
        all_subsets.append((sz, res, tuple(elems)))
        
    # Sort by size to find the minimal max row length
    all_subsets.sort(key=lambda x: x[0])
    
    # Greedily cover residues with the smallest subsets
    best_for_residue = [None] * P
    covered_count = 0
    
    for sz, res, elems in all_subsets:
        if best_for_residue[res] is None:
            best_for_residue[res] = elems
            covered_count += 1
            if covered_count == P:
                # All residues covered. The current 'sz' is the minimal possible max row length
                # because we processed in increasing order of size.
                # Any residue not covered yet would require a larger size, but we just finished covering all.
                # So the max size used is at most sz.
                break
                
    # Construct the final result
    # If for some reason not all are covered (should not happen), fallback to larger sizes
    # But with 1023 subsets and 101 residues, coverage by size 4 or 5 is near-certain.
    # We ensure we have a valid list for every residue.
    result = [None] * P
    for r in range(P):
        if best_for_residue[r] is not None:
            result[r] = list(best_for_residue[r])
        else:
            # Fallback: find any subset that sums to r
            # This should not be reached if covered_count == P
            for sz, res, elems in all_subsets:
                if res == r:
                    result[r] = list(elems)
                    break
            if result[r] is None:
                # Absolute fallback (should be impossible)
                result[r] = [1]
                
    return result
