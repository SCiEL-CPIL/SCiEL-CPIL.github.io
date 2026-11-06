"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard (though this algorithm is O(1024) and instant)
    deadline = time.time() + 59.0
    
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # best_size[r] stores the minimum length of a subset summing to r
    # best_terms[r] stores the subset (as a list) that achieves this minimum
    best_size = [11] * 101  # 11 is larger than max possible size 10
    best_terms = [None] * 101
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    # mask bit i (0-indexed) corresponds to denominator i+1
    for mask in range(1, 1 << 10):
        # Check time limit, though unlikely to be hit
        if time.time() > deadline:
            break
            
        current_sum = 0
        size = 0
        terms = []
        
        # Calculate sum and size for this subset
        # Unrolled or simple loop is fine for 10 items
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                current_sum += inv[i]
                size += 1
                terms.append(i)
        
        r = current_sum % 101
        
        # Update if this subset is smaller than the current best for residue r
        if size < best_size[r]:
            best_size[r] = size
            best_terms[r] = terms
            
    # Construct the result list
    # If a residue was not covered (should not happen), fallback to empty or any valid set
    # But the problem states every residue must be represented, and seed worked, so it's fine.
    result = []
    for r in range(101):
        if best_terms[r] is None:
            # Fallback: find any subset for r (should not occur)
            # Re-scan just in case
            for mask in range(1, 1 << 10):
                s = 0
                t = []
                for i in range(1, 11):
                    if mask & (1 << (i - 1)):
                        s += inv[i]
                        t.append(i)
                if s % 101 == r:
                    best_terms[r] = t
                    break
        result.append(best_terms[r])
        
    return result
# EVOLVE-BLOCK-END
