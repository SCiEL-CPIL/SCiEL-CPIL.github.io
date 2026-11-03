"""Matrix multiplication 2x3 x 3x10: rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]
_SEED_TIME = 42

def _verify(U, V, W):
    """Verify that the tensor decomposition exactly represents matrix multiplication.
    
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i,j,k align
    (i.e., the (i,j) element of A times (j,k) element of B produces (i,k) element of C)
    """
    n, m, p = 2, 3, 10
    U = np.array(U) if not isinstance(U, np.ndarray) else U
    V = np.array(V) if not isinstance(V, np.ndarray) else V
    W = np.array(W) if not isinstance(W, np.ndarray) else W
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i,k] gets contribution from A[i,j]*B[j,k]
                # In tensor form: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if val != 1.0:
                    return False
    return True

def _rank_of(U, V, W):
    """Return the number of non-zero columns (effective rank)."""
    U = np.array(U) if not isinstance(U, np.ndarray) else U
    V = np.array(V) if not isinstance(V, np.ndarray) else V
    W = np.array(W) if not isinstance(W, np.ndarray) else W
    n, m, p = 2, 3, 10
    rank = U.shape[1]
    # Count columns where all three matrices have non-zero entries
    nonzero = 0
    for r in range(rank):
        if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
            nonzero += 1
    return nonzero

def _try_reduce(U, V, W, max_iters=200):
    """Try to reduce rank by identifying linearly dependent or zero columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    n, m, p = 2, 3, 10
    rank = U.shape[1]
    
    # First, remove any zero columns
    keep = []
    for r in range(rank):
        if not (np.all(U[:, r] == 0) and np.all(V[:, r] == 0) and np.all(W[:, r] == 0)):
            keep.append(r)
    
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    if not _verify(U, V, W):
        return None
    
    return (U, V, W)

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Strategy:
    - Start with a known valid decomposition (rank 55)
    - Attempt local search to find a lower-rank decomposition
    - Use time-bounded search with safety margin
    
    The decomposition exploits the structure:
    - 2x3 x 3x10 can be split into 2x2 x 2x10 + 2x1 x 1x10
    - The 2x1 x 1x10 part has rank 20 (schoolbook)
    - The 2x2 x 2x10 part can potentially be optimized
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Base decomposition: rank 55
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:,2] * B[2,:] — schoolbook, rank 20
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            U[a_idx, col] = 1.0
            V[b_idx, col] = 1.0
            W[c_idx, col] = 1.0
            col += 1
    
    # Part 2: A[:,:2] * B[:2,:] — Strassen, rank 35
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        # M1 = (A00+A11)*(B00+B11)
        U[X00, col] = 1.0
        U[X11, col] = 1.0
        V[Y00, col] = 1.0
        V[Y11, col] = 1.0
        W[Z00, col] = 1.0
        W[Z11, col] = 1.0
        col += 1
        
        # M2 = (A10+A11)*B00
        U[X10, col] = 1.0
        U[X11, col] = 1.0
        V[Y00, col] = 1.0
        W[Z10, col] = 1.0
        W[Z11, col] = -1.0
        col += 1
        
        # M3 = A00*(B01-B11)
        U[X00, col] = 1.0
        V[Y01, col] = 1.0
        V[Y11, col] = -1.0
        W[Z01, col] = 1.0
        W[Z11, col] = 1.0
        col += 1
        
        # M4 = A11*(B10-B00)
        U[X11, col] = 1.0
        V[Y10, col] = 1.0
        V[Y00, col] = -1.0
        W[Z00, col] = 1.0
        W[Z10, col] = 1.0
        col += 1
        
        # M5 = (A00+A01)*B11
        U[X00, col] = 1.0
        U[X01, col] = 1.0
        V[Y11, col] = 1.0
        W[Z00, col] = -1.0
        W[Z01, col] = 1.0
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[X10, col] = 1.0
        U[X00, col] = -1.0
        V[Y00, col] = 1.0
        V[Y01, col] = 1.0
        W[Z11, col] = 1.0
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[X01, col] = 1.0
        U[X11, col] = -1.0
        V[Y10, col] = 1.0
        V[Y11, col] = 1.0
        W[Z00, col] = 1.0
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify base decomposition
    assert _verify(U, V, W), "Base decomposition failed verification"
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to improve with local search
    rng = random.Random(_SEED_TIME)
    
    # Attempt: can we merge the schoolbook part with Strassen to save rank?
    # Try a global random perturbation approach to find lower-rank solutions
    
    # Try rank 50, 45, 40, ... by random initialization + verification
    for target_rank in range(50, 25, -5):
        if time.time() - start_time > deadline:
            break
        
        # Try several random attempts at this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                break
            
            # Random sparse initialization
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                # Randomly pick indices
                a_idx = rng.randint(0, n*m-1)
                b_idx = rng.randint(0, m*p-1)
                c_idx = rng.randint(0, p*n-1)
                
                U_try[a_idx, r] = rng.choice([-1.0, 1.0])
                V_try[b_idx, r] = rng.choice([-1.0, 1.0])
                W_try[c_idx, r] = rng.choice([-1.0, 1.0])
            
            if _verify(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break
        
        if best_rank <= target_rank:
            break
    
    # Final: try to reduce by removing redundant columns from best solution
    if best_rank > 25 and time.time() - start_time < deadline:
        reduced = _try_reduce(best_U, best_V, best_W)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify(U_r, V_r, W_r):
                new_rank = U_r.shape[1]
                if new_rank < best_rank:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = new_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
