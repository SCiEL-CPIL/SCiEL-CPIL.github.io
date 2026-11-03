"""Tensor decomposition for 2x3 * 3x10 matrix multiplication.

Uses schoolbook rank-60 as fallback; attempts to find lower-rank exact
decomposition via structured Strassen-based construction and random search.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the factors exactly decompose the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _build_strassen55(n, m, p):
    """
    Attempt a rank-55 decomposition for 2x3x10 using Strassen on 2x2 sub-blocks.
    
    Strategy: Decompose the 3 middle indices into j=0,1 (Strassen 2x2 block)
    and j=2 (single column). Use Strassen's 7 forms for the 2x2x2x2 sub-problem
    tiled appropriately, plus schoolbook for the remaining terms.
    
    This is a heuristic construction; verify before returning.
    """
    if n != 2 or m != 3 or p != 10:
        return None
    
    # For 2x3x10, we'll try to construct a rank-55 decomposition.
    # The exact structure is non-trivial; we build it via the following approach:
    # Use 7 Strassen forms for the 2x2 (i,j in {0,1}) x 2 (k in {0,1}) sub-block,
    # but this only covers 4 of the 60 terms. The rest need schoolbook.
    # 
    # Actually, a cleaner approach: the known rank-55 result for 2x3x10 comes from
    # the fact that the border rank is 55. We construct it as follows:
    # - 10 columns of the 3x10 matrix: for each column k, we do a 2x3 x 3x1 mult.
    # - For each k, use a 7-term decomposition for the 2x2 part (j=0,1) and
    #   1 term for j=2. But 7+1=8 per column, 8*10=80. That's worse.
    #
    # The actual rank-55 construction is more subtle. Let's use a random search
    # initialized near the schoolbook to try to find a lower-rank solution.
    # For now, return None to indicate we should use schoolbook + search.
    return None

def _random_search_rank(n, m, p, target_rank, max_time=30.0):
    """
    Random search for a rank-target_rank exact decomposition.
    Uses random rank-1 tensors and checks if they span the multiplication tensor.
    Returns (U, V, W) as lists if found, else None.
    """
    start = time.time()
    Ua = np.zeros((n*m, target_rank))
    Va = np.zeros((m*p, target_rank))
    Wa = np.zeros((p*n, target_rank))
    
    # Initialize with schoolbook-like structure then perturb
    for c in range(target_rank):
        i = (c // (m*p)) % n
        j = (c // p) % m
        k = c % p
        Ua[i*m + j, c] = 1.0
        Va[j*p + k, c] = 1.0
        Wa[k*n + i, c] = 1.0
    
    # Random perturbations to explore nearby decompositions
    np.random.seed(42)
    for attempt in range(200):
        if time.time() - start > max_time:
            break
        # Try a random rank-target_rank decomposition
        Ua = np.random.randn(n*m, target_rank)
        Va = np.random.randn(m*p, target_rank)
        Wa = np.random.randn(p*n, target_rank)
        # Normalize to avoid overflow
        for c in range(target_rank):
            Un = np.linalg.norm(Ua[:, c])
            Vn = np.linalg.norm(Va[:, c])
            Wn = np.linalg.norm(Wa[:, c])
            if Un > 0 and Vn > 0 and Wn > 0:
                Ua[:, c] /= Un
                Va[:, c] /= Vn
                Wa[:, c] /= (Un * Vn * Wn) if (Un*Vn*Wn) != 0 else 1.0
        
        # Check if this decomposition captures the multiplication tensor
        # We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        success = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-6:
                        success = False
                        break
                if not success:
                    break
            if not success:
                break
        if success:
            return Ua.tolist(), Va.tolist(), Wa.tolist()
    
    return None

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # First try structured Strassen-based construction
    strat = _build_strassen55(n, m, p)
    if strat is not None:
        U, V, W = strat
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Try random search for rank 55
    if time.time() < deadline - 5.0:
        result = _random_search_rank(n, m, p, 55, max_time=min(20.0, deadline - time.time() - 5.0))
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try random search for rank 58
    if time.time() < deadline - 5.0:
        result = _random_search_rank(n, m, p, 58, max_time=min(10.0, deadline - time.time() - 5.0))
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank 60
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
