import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    p = 101
    denominators = list(range(1, 11))
    n = 10
    
    # Precompute modular inverses
    inverses = [0] * (n + 1)
    for x in denominators:
        inverses[x] = pow(x, -1, p)
    
    # Precompute all subsets grouped by size
    # subsets_by_size[k] = list of (residue, combo) for subsets of size k
    subsets_by_size = {}
    for size in range(1, n + 1):
        subsets_by_size[size] = []
        for combo in itertools.combinations(denominators, size):
            s = sum(inverses[d] for d in combo)
            subsets_by_size[size].append((s % p, combo))
    
    # Binary search on maximum allowed length k
    # Check if all residues can be covered using subsets of size <= k
    def can_cover(max_k):
        covered = set()
        for k in range(1, max_k + 1):
            for residue, combo in subsets_by_size[k]:
                covered.add(residue)
            if len(covered) == p:
                return True
        return len(covered) == p
    
    # Find minimum k
    lo, hi = 1, n
    min_k = n
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_cover(mid):
            min_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Assign each residue the smallest valid subset (size <= min_k)
    # Process by size to ensure smallest assignment
    assignments = {}
    for k in range(1, min_k + 1):
        for residue, combo in subsets_by_size[k]:
            if residue not in assignments:
                assignments[residue] = combo
        if len(assignments) == p:
            break
    
    # Time guard
    if time.time() > deadline - 5.0:
        # Fallback: just return what we have
        result = []
        for r in range(p):
            if r in assignments:
                result.append(list(assignments[r]))
            else:
                result.append(denominators[:])
        return result
    
    # Construct the result
    result = []
    for r in range(p):
        if r in assignments:
            result.append(list(assignments[r]))
        else:
            # Should not happen
            result.append(denominators[:])
            
    return result
