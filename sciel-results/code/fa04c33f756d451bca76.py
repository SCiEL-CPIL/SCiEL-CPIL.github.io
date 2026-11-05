"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit for safety
    
    # Precompute inverses for 1..10 modulo 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # We want to find the minimum size subset for each residue
    # Initialize with a size larger than any possible subset (10)
    min_size = [11] * 101
    best_subset = [None] * 101
    
    # Iterate through all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
            
        terms = []
        residue = 0
        size = 0
        for i in range(10):
            if mask & (1 << i):
                val = i + 1
                terms.append(val)
                residue = (residue + inv[val]) % 101
                size += 1
        
        # If this subset is smaller than the current best for this residue, update
        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = terms
            
    # Construct the result
    result = []
    max_len = 0
    
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
            if len(best_subset[r]) > max_len:
                max_len = len(best_subset[r])
        else:
            # Fallback: This should not happen if coverage is complete.
            # If it does, we have a problem. 
            # To satisfy "nonempty lists", we must find *some* subset.
            # But if min_size is 11, no subset was found.
            # This implies the residue is not covered by any subset of 1..10.
            # Given the problem statement "Every residue must be represented", 
            # this case should be impossible for the given parameters.
            # However, to avoid returning an empty list (invalid), 
            # we might need a fallback strategy, but there is none if no subset exists.
            # We assume it is covered.
            result.append([])
            
    return result
# EVOLVE-BLOCK-END
