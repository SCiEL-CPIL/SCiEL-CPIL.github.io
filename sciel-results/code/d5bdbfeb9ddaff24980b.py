"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # Constants
    P = 101
    N = 10
    DEADLINE = 80  # Safety margin before 90s kill
    
    # Pre-compute modular inverses for 1..10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)
    
    # Helper to convert mask to list of denominators
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << (i - 1))]
    
    # Minimum-cost bottleneck assignment:
    # Since denominators may recur across rows (only distinct within a row),
    # each residue independently picks its minimum-size valid subset.
    # The objective (max row length) = max over residues of min-size subset for that residue.
    # We compute min-size subset per residue, then verify coverage.
    
    # min_size[r] = smallest size of any subset summing to r mod P
    # best_mask[r] = a mask achieving that minimum size
    min_size = [N + 1] * P  # initialized to impossible
    best_mask = [0] * P
    
    # Enumerate all non-empty subsets of {1..N}
    # For efficiency, iterate by size and stop early once all residues are covered at current size
    for size in range(1, N + 1):
        if time.time() - start_time > DEADLINE:
            break
        # Generate all masks of exactly this size
        from itertools import combinations
        for combo in combinations(range(N), size):
            mask = 0
            s = 0
            for idx in combo:
                bit = 1 << idx
                mask |= bit
                s = (s + inv[idx + 1]) % P
            if min_size[s] > size:
                min_size[s] = size
                best_mask[s] = mask
    
    # Check if all residues are covered
    uncovered = [r for r in range(P) if min_size[r] > N]
    
    if uncovered:
        # Fallback: this shouldn't happen, but ensure validity
        # Use any subset for uncovered residues
        all_subsets = []
        for mask in range(1, 1 << N):
            s = 0
            for i in range(1, N + 1):
                if mask & (1 << (i - 1)):
                    s = (s + inv[i]) % P
            all_subsets.append((s, mask))
        
        reps = {}
        for s, mask in all_subsets:
            if s not in reps:
                reps[s] = mask
        
        final_result = []
        for r in range(P):
            if r in reps:
                final_result.append(mask_to_list(reps[r]))
            else:
                final_result.append([1])
        return final_result
    
    # Construct result: each residue gets its minimum-size subset
    result = []
    for r in range(P):
        result.append(mask_to_list(best_mask[r]))
    
    return result

# EVOLVE-BLOCK-END
