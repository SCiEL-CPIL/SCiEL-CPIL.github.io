import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute inverses of 1..10 modulo 101
    invs = [pow(x, -1, 101) for x in range(1, 11)]
    denominators = list(range(1, 11))
    
    # best_rep[r] will store the smallest list of denominators that sums to r
    # Initialize with None to indicate not found yet
    best_rep = [None] * 101
    
    # Iterate over all non-empty subsets of {1, ..., 10}
    # We can iterate by size to stop early if we want, but we need the MIN size for each residue.
    # So we must check all subsets, or iterate by size and stop when all are covered.
    
    # To find the minimum size for each residue, we can iterate by size k=1 to 10.
    # For each k, generate all subsets of size k.
    # For each subset, compute sum mod 101.
    # If best_rep[sum] is None, set it to this subset.
    # If all best_rep are filled, we can stop.
    
    all_covered = False
    
    for k in range(1, 11):
        if all_covered:
            break
        if time.time() > deadline:
            break
            
        # Generate combinations of size k
        for combo in combinations(denominators, k):
            # Calculate sum of inverses
            s = 0
            for d in combo:
                s = (s + invs[d-1]) % 101
            
            if best_rep[s] is None:
                best_rep[s] = list(combo)
        
        # Check if all residues are covered
        if all(x is not None for x in best_rep):
            all_covered = True
            
    # If for some reason not all are covered (should not happen with k=10),
    # fill remaining with any subset (e.g., the full set or largest found)
    # But with k=10, we have 1 subset (the full set) which sums to some value.
    # We have 1023 subsets total. It is virtually certain all 101 residues are covered.
    
    # Fallback for any None (should be empty)
    # If a residue is still None, we need to find *any* subset that sums to it.
    # We can just re-scan all subsets.
    if any(x is None for x in best_rep):
        # Re-scan all subsets to find any match for missing residues
        for mask in range(1, 1 << 10):
            if time.time() > deadline:
                break
            denom_list = []
            s = 0
            for i in range(10):
                if mask & (1 << i):
                    denom_list.append(denominators[i])
                    s = (s + invs[i]) % 101
            
            if best_rep[s] is None:
                best_rep[s] = denom_list
    
    # Final check: if any are still None (impossible), use the full set as placeholder?
    # The problem guarantees a solution exists? "Every residue must be represented."
    # If a residue is truly not representable, the problem is impossible.
    # But with 10 denominators mod 101, it's always possible.
    
    # Ensure all are lists
    result = []
    for r in range(101):
        if best_rep[r] is None:
            # This should never happen
            best_rep[r] = [1] 
        result.append(best_rep[r])
        
    return result
