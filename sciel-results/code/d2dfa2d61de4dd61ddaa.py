"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs 101 lists of distinct denominators in 1..10 such that the sum of 
    their modular inverses mod 101 equals the row index.
    Minimizes the maximum length of the lists.
    
    Strategy:
    Since denominators can recur across different rows, the problem decomposes into 
    101 independent sub-problems. For each residue r, we find the subset of 
    denominators with the minimum size that sums to r mod 101.
    The optimal global solution is the collection of these individual minima.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin from 90s limit
    
    MOD = 101
    DENOMS = list(range(1, 11))
    
    # Precompute modular inverses
    invs = [0] * 11
    for x in DENOMS:
        invs[x] = pow(x, -1, MOD)
        
    # min_size_for_res[r] will store the minimum size of a subset summing to r.
    # best_subset_for_res[r] will store the actual subset (list of denominators).
    
    min_size_for_res = [11] * MOD  # 11 is effectively infinity (max size is 10)
    best_subset_for_res = [None] * MOD
    
    # Iterate through all non-empty subsets of DENOMS
    # There are 2^10 - 1 = 1023 such subsets.
    for mask in range(1, 1 << 10):
        current_sum = 0
        current_size = 0
        terms = []
        
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                terms.append(denom)
                current_sum = (current_sum + invs[denom]) % MOD
                current_size += 1
        
        residue = current_sum
        
        # Update if this subset is smaller than the current best for this residue
        if current_size < min_size_for_res[residue]:
            min_size_for_res[residue] = current_size
            best_subset_for_res[residue] = terms
            
        # Deadline check
        if time.time() - start_time > DEADLINE:
            break

    # Construct the result
    result = []
    for r in range(MOD):
        if best_subset_for_res[r] is None:
            # Fallback: This should not happen if all residues are reachable by non-empty subsets.
            # If it does happen, we return an empty list (which violates "nonempty" but is better than crashing).
            # However, based on previous runs, this case likely doesn't occur.
            result.append([])
        else:
            result.append(best_subset_for_res[r])
            
    return result
# EVOLVE-BLOCK-END
