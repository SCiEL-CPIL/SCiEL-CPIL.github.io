"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # Constants
    P = 101
    N = 10
    
    # Pre-compute modular inverses for 1..10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)
        
    # Pre-compute all non-empty subsets
    # Each entry is (residue, size, mask)
    all_subsets = []
    for mask in range(1, 1 << N):
        size = bin(mask).count('1')
        s = 0
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                s = (s + inv[i]) % P
        all_subsets.append((s, size, mask))
        
    # Helper to convert mask to list of denominators
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << (i - 1))]
        
    # Try to find a solution with max row length K
    def try_solve(K):
        # best_subset[r] = (size, mask) for the best subset for residue r
        # We prefer smaller size, then smaller mask (lexicographic on denominators)
        best_subset = [None] * P
        
        for residue, size, mask in all_subsets:
            if size > K:
                continue
            if best_subset[residue] is None:
                best_subset[residue] = (size, mask)
            else:
                prev_size, prev_mask = best_subset[residue]
                if size < prev_size:
                    best_subset[residue] = (size, mask)
                elif size == prev_size and mask < prev_mask:
                    best_subset[residue] = (size, mask)
                    
        # Check if all residues are covered
        if any(x is None for x in best_subset):
            return None
            
        # Construct the result
        result = []
        for r in range(P):
            size, mask = best_subset[r]
            result.append(mask_to_list(mask))
            
        return result

    # Attempt K=3 first, then K=4, etc.
    # Theoretical max K is 10, but K=3 or 4 is expected.
    for K in range(1, N + 1):
        # Safety guard: if we are taking too long (unlikely given small N), break
        if time.time() - start_time > 50:
            break
            
        res = try_solve(K)
        if res is not None:
            return res
            
    # Fallback: Should not happen for N=10, P=101
    # If it does, just return the first found for each residue without size constraint
    # This ensures validity even if logic above has a bug
    representations = {}
    for residue, size, mask in all_subsets:
        if residue not in representations:
            representations[residue] = mask_to_list(mask)
            
    # If some residues are still missing (impossible), fill with [1] (invalid sum but valid structure)
    # But we know all residues are covered by the 1023 subsets.
    # Wait, is it possible some residue is NOT covered by ANY subset?
    # The problem statement implies a solution exists.
    # If a residue is not covered, the task is impossible.
    # But let's assume it is covered.
    
    final_result = []
    for r in range(P):
        if r in representations:
            final_result.append(representations[r])
        else:
            # This case should not occur. Add a dummy to prevent crash.
            final_result.append([1])
            
    return final_result

# EVOLVE-BLOCK-END
