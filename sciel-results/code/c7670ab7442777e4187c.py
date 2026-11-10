import time

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin for 60s limit
    
    p = 101
    # Precompute inverses of 1..10 modulo 101
    inverses = [0] * 11
    for i in range(1, 11):
        inverses[i] = pow(i, -1, p)
        
    # best_subset[r] will store the subset (as a list of denominators) with minimum length
    # that sums to r mod 101.
    best_subset = [None] * p
    best_len = [float('inf')] * p
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    for mask in range(1, 1 << 10):
        # Check time limit
        if time.time() > deadline:
            break
            
        terms = []
        s = 0
        # Unroll loop for speed? 10 iterations is fine.
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                terms.append(i)
                s = (s + inverses[i]) % p
        
        length = len(terms)
        residue = s
        
        if length < best_len[residue]:
            best_len[residue] = length
            best_subset[residue] = terms
        
    # Construct the result list
    result = []
    for r in range(p):
        if best_subset[r] is None:
            # Fallback, though should not happen
            best_subset[r] = []
        result.append(best_subset[r])
        
    return result
