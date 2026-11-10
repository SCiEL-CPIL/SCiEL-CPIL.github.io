"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # Precompute modular inverses of 1..10 modulo 101
    # invs[x] is the modular inverse of x modulo 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)
        
    # best_subset[r] will store the list of terms for residue r
    # best_len[r] will store the length of the best subset found so far
    # Initialize with infinity (or a large number)
    best_subset = [None] * 101
    best_len = [11] * 101  # 11 is larger than max possible length 10
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    # We can iterate by size to potentially stop early if we find size 1 or 2,
    # but since we need to check all to find the absolute minimum for each residue,
    # and the total number is small (1023), we just iterate all masks.
    # However, iterating by mask allows us to compute the sum efficiently if we use DP,
    # but 1023 is small enough for direct calculation.
    
    for mask in range(1, 1 << 10):
        # Calculate sum and length
        current_sum = 0
        current_terms = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                current_sum += invs[x]
                current_terms.append(x)
        
        residue = current_sum % 101
        length = len(current_terms)
        
        # Update if this subset is shorter than the current best for this residue
        if length < best_len[residue]:
            best_len[residue] = length
            best_subset[residue] = current_terms
            
        # Safety check for time (though 1023 iterations is instantaneous)
        if time.time() - start_time > 85:
            break

    # Construct the final list
    # If any residue is not covered (best_subset[r] is None), 
    # it means no subset sums to r, which is impossible given the problem statement implies a solution exists.
    # But to be safe, if None, we can try to find any subset or return an empty list (though task says nonempty).
    # Given 1023 subsets and 101 residues, coverage is virtually certain.
    
    result = []
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: This should not happen. 
            # If it does, we might need a more complex search or the problem is infeasible.
            # But for p=101 and denom 1..10, it is feasible.
            # We will append an empty list to avoid crash, but the evaluator might fail us.
            result.append([])
            
    return result
# EVOLVE-BLOCK-END
