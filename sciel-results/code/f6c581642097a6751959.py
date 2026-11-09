"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    p = 101
    # Precompute inverses for denominators 1..10 modulo 101
    invs = [pow(x, -1, p) for x in range(1, 11)]
    
    # We need to find, for each residue r in 0..100, a non-empty subset of {1..10}
    # whose inverse-sum mod 101 equals r.
    # Minimize the maximum subset size (i.e., minimize the largest row length).
    
    # Strategy: For each k from 1 to 10, check if all 101 residues can be covered
    # by subsets of size exactly k (or <= k). We want the smallest k such that
    # every residue has at least one subset of size <= k that sums to it.
    # Then assign each residue the smallest valid subset (preferring smaller size).
    
    # Precompute all subset sums grouped by size
    # subset_sums[size][residue] = list of (size, terms) that achieve that residue
    # Actually, let's just collect for each residue the best (smallest) subset
    
    # For each residue, track the best (smallest) subset found so far
    best_for_residue = {r: None for r in range(p)}
    
    # Enumerate subsets by size from 1 to 10
    # For each size k, enumerate all C(10, k) subsets
    for k in range(1, 11):
        if time.time() > deadline:
            break
        for combo in combinations(range(10), k):
            if time.time() > deadline:
                break
            residue = 0
            for i in combo:
                residue = (residue + invs[i]) % p
            terms = [i + 1 for i in combo]
            
            # If this residue hasn't been assigned yet, assign it
            if best_for_residue[residue] is None:
                best_for_residue[residue] = (k, terms)
    
    # Check if all residues are covered
    uncovered = [r for r in range(p) if best_for_residue[r] is None]
    
    if uncovered:
        # If some residues are still uncovered (shouldn't happen with k up to 10),
        # we need a fallback. Let's try a more exhaustive approach.
        # Actually, with 10 elements we have 1023 non-empty subsets, which should
        # cover all 101 residues. But let's be safe.
        # Try a greedy/DP approach for uncovered residues
        for r in uncovered:
            if time.time() > deadline:
                break
            # Brute force: try all subsets again for this specific residue
            for mask in range(1, 1 << 10):
                terms = []
                residue = 0
                for i in range(10):
                    if mask & (1 << i):
                        terms.append(i + 1)
                        residue = (residue + invs[i]) % p
                if residue == r:
                    best_for_residue[r] = (len(terms), terms)
                    break
    
    # Build the result
    result = []
    for r in range(p):
        if best_for_residue[r] is not None:
            size, terms = best_for_residue[r]
            result.append(terms)
        else:
            # This should not happen, but as a fallback, use a single element
            # (which won't sum to r, but we need to return something)
            # Actually, let's try to find ANY subset that sums to r
            result.append([1])  # fallback, shouldn't be reached
    
    return result
# EVOLVE-BLOCK-END
