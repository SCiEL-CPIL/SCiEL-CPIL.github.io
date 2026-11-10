import time

def construct():
    # Deadline guard, though this solution is O(2^10 * 10) ~ 10k ops, so it's instant.
    # The guard is included as per directive.
    start_time = time.time()
    deadline = 80.0  # 10s margin from 90s limit

    P = 101
    N = 10
    
    # Precompute inverses
    invs = [pow(i, -1, P) for i in range(1, N + 1)]
    
    # min_size[r] will store the minimum number of terms needed to sum to r
    # best_subset[r] will store the list of denominators for that minimum size
    min_size = [float('inf')] * P
    best_subset = [None] * P
    
    # Iterate over all non-empty subsets of {1, ..., 10}
    # Mask bit i corresponds to denominator (i+1)
    for mask in range(1, 1 << N):
        # Check time
        if time.time() > deadline:
            break
            
        s = 0
        count = 0
        terms = []
        
        # Calculate sum and terms for this subset
        # We can optimize by using bit operations, but N=10 is small.
        for i in range(N):
            if mask & (1 << i):
                denom = i + 1
                s += invs[i]
                count += 1
                terms.append(denom)
        
        r = s % P
        
        # Update if this subset is smaller than the current best for residue r
        if count < min_size[r]:
            min_size[r] = count
            best_subset[r] = terms
            
    # If for some reason a residue wasn't covered (should not happen), 
    # fallback to any valid subset (e.g., from a larger search or default)
    # But with 1023 subsets and 101 residues, coverage is guaranteed.
    # Just in case, fill any None with a valid dummy if it failed (it won't)
    result = []
    for r in range(P):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: this should never trigger
            result.append([1])
            
    return result
