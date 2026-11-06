import time
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # Precompute all subsets by size and their residue sums
    # subsets_by_size[size] = list of (residue, subset_tuple)
    subsets_by_size = {s: [] for s in range(1, 11)}
    
    for size in range(1, 11):
        for combo in combinations(D, size):
            res = sum(inv[x] for x in combo) % P
            subsets_by_size[size].append((res, combo))
    
    # For each size limit L, get all subsets of size <= L
    # We want to find minimum L such that we can assign one subset to each residue
    
    # Strategy: For a given L, we have a pool of subsets (size <= L).
    # We need to assign each residue r a subset from the pool that sums to r.
    # Multiple residues can use the same subset (denominators can recur across rows).
    # So this is simply: for each residue, is there at least one subset in the pool that hits it?
    
    # Wait - re-reading the problem: "Denominators may recur across different rows, but never within one row."
    # This means each row is an independent choice. So for each residue r, we just need to find ANY subset
    # (of size <= L) whose inverse sum is r mod 101.
    
    # So the problem is much simpler than I thought! For each residue, independently find the smallest subset.
    # The maximum row length is just the maximum over all residues of the minimum subset size that hits it.
    
    # Let me verify: the current program does exactly this greedy approach.
    # So the question is: what is the actual minimum maximum size?
    
    # For each residue, find the minimum size subset that hits it
    min_size_for_residue = [11] * P  # 11 means not found yet
    
    for size in range(1, 11):
        for res, combo in subsets_by_size[size]:
            if min_size_for_residue[res] > size:
                min_size_for_residue[res] = size
    
    # Find the maximum minimum size
    max_min_size = max(min_size_for_residue)
    
    # Now assign for each residue the smallest subset
    result = [None] * P
    for r in range(P):
        target_size = min_size_for_residue[r]
        for res, combo in subsets_by_size[target_size]:
            if res == r:
                result[r] = list(combo)
                break
        # If not found (shouldn't happen), fallback
        if result[r] is None:
            for s in range(target_size + 1, 11):
                for res, combo in subsets_by_size[s]:
                    if res == r:
                        result[r] = list(combo)
                        break
                if result[r] is not None:
                    break
    
    return result
