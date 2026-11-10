"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

# EVOLVE-BLOCK-START
def _sum_counts(mask, N):
    """Return array c where c[s] = number of pairs (a,b) with a<=b, a,b in mask, a+b=s.
    mask: boolean array of length N+1 (indices 1..N)
    """
    # Use convolution of mask with itself
    # mask as float
    m = mask.astype(np.float64)
    # convolve
    conv = np.convolve(m, m, mode='full')
    # conv[k] = sum_i m[i]*m[k-i]
    # For sum s = a+b, a<=b: count = (conv[s] + (1 if s even and mask[s//2] else 0)) / 2
    # Actually: conv[s] = sum_{a+b=s} m[a]m[b] = 2*(#pairs a<b) + (#pairs a=b)
    # #pairs a<=b = (#pairs a<b) + (#pairs a=b) = (conv[s] - diag)/2 + diag = (conv[s] + diag)/2
    # where diag = 1 if s even and mask[s//2] else 0
    c = np.zeros(N + 1, dtype=np.int32)
    for s in range(2, N + 1):
        diag = 1 if (s % 2 == 0 and mask[s // 2]) else 0
        c[s] = (conv[s] + diag) // 2
    return c

def construct():
    N = 1000
    random.seed(42)
    np.random.seed(42)
    
    # Dense random initialization: ~700 integers
    target_size = 700
    pool = list(range(1, N + 1))
    random.shuffle(pool)
    chosen = pool[:target_size]
    
    def is_valid(mask):
        c = _sum_counts(mask, N)
        return np.max(c) <= 2
    
    def collision_excess(mask):
        c = _sum_counts(mask, N)
        return np.sum(np.maximum(c - 2, 0))
    
    # Build initial mask
    mask = np.zeros(N + 1, dtype=bool)
    for x in chosen:
        mask[x] = True
    
    current_excess = collision_excess(mask)
    
    # If valid, we can try to add more
    if current_excess == 0:
        # Greedily add more elements
        for x in range(1, N + 1):
            if not mask[x]:
                mask[x] = True
                if np.max(_sum_counts(mask, N)) <= 2:
                    chosen.append(x)
                else:
                    mask[x] = False
        return chosen
    
    # Local search: iteratively try swaps to reduce excess
    best_excess = current_excess
    best_mask = mask.copy()
    
    # Precompute for speed
    no_improve = 0
    max_iters = 5000
    
    for it in range(max_iters):
        if no_improve > 500:
            break
        # Pick a random element to remove
        in_set = np.where(mask)[0]
        if len(in_set) == 0:
            break
        a = in_set[random.randint(0, len(in_set) - 1)]
        # Pick a random element to add
        not_in_set = np.where(~mask)[0]
        if len(not_in_set) == 0:
            break
        b = not_in_set[random.randint(0, len(not_in_set) - 1)]
        
        # Try swap a -> b
        mask[a] = False
        mask[b] = True
        new_excess = collision_excess(mask)
        
        if new_excess < best_excess:
            best_excess = new_excess
            best_mask = mask.copy()
            no_improve = 0
        else:
            # Revert
            mask[b] = False
            mask[a] = True
            no_improve += 1
    
    # Extract result from best_mask
    result = np.where(best_mask)[0].tolist()
    return result
# EVOLVE-BLOCK-END
