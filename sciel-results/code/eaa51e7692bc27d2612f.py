"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Trivial schoolbook decomposition (rank = n*m*p = 105)
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((mp, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((pn, rank_trivial), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1
                V_trivial[b, col] = 1
                W_trivial[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_computed, T_target, atol=1e-9)
    
    # Lower bound is max(nm, mp, pn) = 35
    lower_bound = max(nm, mp, pn)
    
    # Strategy: Try structured decompositions
    # For 3x5x7, try to exploit the small n=3
    
    # Approach 1: Try to find a decomposition using the structure
    # where for each j, we have a block that handles all (i,k) pairs
    
    # Approach 2: Random search with better initialization
    
    rng = np.random.default_rng(42)
    
    # Try ranks from a reasonable range
    # Start from lower_bound and go up, but focus on promising ranges
    # Known results suggest rank might be around 50-70 for these dimensions
    
    # Try several strategies
    strategies = []
    
    # Strategy 1: Random with {-1,0,1} entries, varying density
    for density in [0.3, 0.4, 0.5]:
        strategies.append(('random', density))
    
    # Strategy 2: Block structure - for each j, use a block of size n*p
    # This gives rank = m * (n*p) = 5*21 = 105, same as schoolbook
    # But maybe we can do better with shared blocks
    
    # Strategy 3: Try to merge terms from different j's
    
    # Let's try a more systematic approach:
    # For each target rank, try multiple random initializations
    
    # Focus on ranks that might be achievable
    # Lower bound is 35, but realistically we need more
    # Try ranks from 45 to best_rank-1
    
    target_ranks = list(range(max(lower_bound + 5, 45), best_rank))
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # More trials for lower ranks
        max_trials = 100 if target_rank < 60 else 50
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with entries from {-1, 0, 1}
            # Use varying densities
            density = 0.3 + 0.1 * (trial % 3)
            
            U = rng.integers(-1, 2, size=(nm, target_rank)).astype(np.float64)
            V = rng.integers(-1, 2, size=(mp, target_rank)).astype(np.float64)
            W = rng.integers(-1, 2, size=(pn, target_rank)).astype(np.float64)
            
            # Apply sparsity
            for mat in [U, V, W]:
                mask = rng.random(mat.shape) < density
                mat[~mask] = 0
            
            if verify(U, V, W):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break
        
        # If we found a decomposition at this rank, continue to try lower
        # but we're going from low to high, so we can stop if we want
        # Actually, we're going from low to high, so first success is best
        if best_rank == target_rank:
            break
    
    # If we didn't find anything better, try a different approach:
    # Use the schoolbook decomposition but try to reduce it
    
    if best_rank == rank_trivial and time.time() < deadline - 30:
        # Try to find a lower rank by solving a system of equations
        # This is harder, so let's try a different random strategy
        
        # Use entries from {-1, 0, 1, 2} with lower density
        for target_rank in range(max(lower_bound + 10, 50), best_rank):
            if time.time() > deadline - 15:
                break
            
            for trial in range(30):
                if time.time() > deadline - 10:
                    break
                
                U = rng.integers(-2, 3, size=(nm, target_rank)).astype(np.float64)
                V = rng.integers(-2, 3, size=(mp, target_rank)).astype(np.float64)
                W = rng.integers(-2, 3, size=(pn, target_rank)).astype(np.float64)
                
                # Apply sparsity
                for mat in [U, V, W]:
                    mask = rng.random(mat.shape) < 0.4
                    mat[~mask] = 0
                
                if verify(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
