import time
from itertools import combinations
from collections import deque

def construct():
    deadline = time.time() + 85.0
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
        
    # Generate all subsets
    subsets = []
    for size in range(1, 11):
        for comb in combinations(range(1, 11), size):
            s = sum(inv[x] for x in comb) % p
            subsets.append((size, list(comb), s))
            
    # Sort by size
    subsets.sort(key=lambda x: x[0])
    
    # We want to find the min K such that there is a matching covering all residues using subsets of size <= K.
    # Since each subset maps to a unique residue, the matching is trivial.
    # But let's implement the check properly.
    
    # For each K, build graph with edges (r, subset_idx) where subset size <= K and sum = r.
    # Check if max matching size is 101.
    
    # Optimization: Since subsets are unique per residue, we can just check if for every residue r,
    # there exists at least one subset of size <= K summing to r.
    # This is equivalent to the independent minimization.
    
    # So I will stick to the independent minimization as it is O(1) per subset.
    
    min_len = [11] * p
    best_subsets = [None] * p
    
    for size, comb, s in subsets:
        if size < min_len[s]:
            min_len[s] = size
            best_subsets[s] = comb
            
    return [best_subsets[r] for r in range(p)]
