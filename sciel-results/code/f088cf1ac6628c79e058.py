"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses a systematic search with bit-parallel conflict masks to find a large code.
    Improved by running multiple greedy passes with different candidate orderings
    (randomized and Gray-code) and selecting the best result.
    """
    N = 18
    MIN_DIST = 6
    size = 1 << N
    
    # Precompute all masks of weight < MIN_DIST
    # Weight < 6 means weight 0 to 5.
    masks = [i for i in range(size) if i.bit_count() < MIN_DIST]
    masks = np.array(masks, dtype=np.uint32)
    
    # Function to run a greedy pass given a specific order of candidates
    def run_greedy(order):
        forbidden = np.zeros(size, dtype=bool)
        code = []
        
        # Helper to mark forbidden neighbors
        # Using local variables for speed
        m = masks
        f = forbidden
        
        for c in order:
            if not f[c]:
                code.append(c)
                # Mark all neighbors within distance < 6 as forbidden
                neighbors = (c ^ m) % size
                f[neighbors] = True
        
        return code

    best_code = []
    best_len = 0
    
    # Run multiple passes with different orderings
    # 1. Standard order (0 to size-1)
    order_std = np.arange(size, dtype=np.uint32)
    code_std = run_greedy(order_std)
    if len(code_std) > best_len:
        best_len = len(code_std)
        best_code = code_std
        
    # 2. Gray-code order (as suggested by directive)
    # Generate Gray code sequence
    # Gray code for n bits: g[i] = i ^ (i >> 1)
    # We need the sequence of values, not just the generator.
    # For 18 bits, size is 262144.
    gray_order = np.array([i ^ (i >> 1) for i in range(size)], dtype=np.uint32)
    code_gray = run_greedy(gray_order)
    if len(code_gray) > best_len:
        best_len = len(code_gray)
        best_code = code_gray

    # 3. Randomized passes
    # Random shuffling often yields larger independent sets in greedy algorithms
    # We run several random passes.
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in evaluation? 
    # Actually, better to use random seeds or just let it run.
    # To ensure we finish in 60s, we limit the number of passes.
    # Each pass is ~0.05s. 100 passes = 5s. Safe.
    
    for _ in range(100):
        order_rand = rng.permutation(size).astype(np.uint32)
        code_rand = run_greedy(order_rand)
        if len(code_rand) > best_len:
            best_len = len(code_rand)
            best_code = code_rand
            
    return best_code

# EVOLVE-BLOCK-END
