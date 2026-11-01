"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the trivial rank-nmp schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact tensor decomposition using vectorized matrix checks."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, -1)
    
    for i in range(n):
        for k in range(p):
            U_i = U_reshaped[i]
            w_idx = k * n + i
            w_row = W[w_idx]
            v_indices = np.arange(k, m*p, p)
            V_k = V[v_indices]
            
            U_scaled = U_i * w_row
            M = U_scaled @ V_k.T
            
            if not np.allclose(M, np.eye(m), atol=tol):
                return False
    return True

def _try_rank(r, n, m, p, deadline, seed_offset=0, max_trials=100):
    """Try to find a valid decomposition of rank r using random search with local refinement."""
    np.random.seed(42 + seed_offset)
    
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values from {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n*m, r))
        V = np.random.choice([-1, 0, 1], size=(m*p, r))
        W = np.random.choice([-1, 0, 1], size=(p*n, r))
        
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        # Try local refinement: flip random entries and check
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            mat_idx = np.random.choice(3)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
            
            idx = np.random.randint(mat.shape[0])
            col = np.random.randint(r)
            old_val = mat[idx, col]
            new_val = np.random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3
            
            mat[idx, col] = new_val
            if _verify(U, V, W, n, m, p):
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            mat[idx, col] = old_val
    
    return None

def _build_rank18_from_2x3x3(U_9, V_9, W_9):
    """
    Construct a rank 18 decomposition for 2x3x6 from a rank 9 decomposition for 2x3x3.
    U_9: (6, 9)
    V_9: (9, 9)
    W_9: (6, 9)
    
    The 2x3x6 multiplication can be seen as two 2x3x3 multiplications side-by-side.
    U' = [U_9, U_9] (6, 18)
    V' = [V_9, 0; 0, V_9] (18, 18) -> Actually V is (m*p, rank) = (18, 18)
    Rows 0-8 correspond to first 3x3 block of B.
    Rows 9-17 correspond to second 3x3 block of B.
    W' = [W_9, W_9] (12, 18) -> Rows 0-5 for k=0..2, Rows 6-11 for k=3..5.
    """
    n, m, p = 2, 3, 6
    rank = 18
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # First block (k=0,1,2)
    U[:, 0:9] = U_9
    V[0:9, 0:9] = V_9
    W[0:6, 0:9] = W_9
    
    # Second block (k=3,4,5)
    U[:, 9:18] = U_9
    V[9:18, 9:18] = V_9
    W[6:12, 9:18] = W_9
    
    return U, V, W

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    # Strategy: Find rank 9 for 2x3x3, then build rank 18 for 2x3x6
    # 2x3x3 dimensions
    n2, m2, p2 = 2, 3, 3
    r_target = 9
    
    # Try to find rank 9 decomposition for 2x3x3
    result_233 = _try_rank(r_target, n2, m2, p2, deadline, seed_offset=100, max_trials=50)
    
    if result_233 is not None:
        U_9 = np.array(result_233['U'])
        V_9 = np.array(result_233['V'])
        W_9 = np.array(result_233['W'])
        
        U_18, V_18, W_18 = _build_rank18_from_2x3x3(U_9, V_9, W_9)
        
        # Verify the constructed 2x3x6 decomposition
        if _verify(U_18, V_18, W_18, n, m, p):
            best_rank = 18
            best_result = {'U': U_18.tolist(), 'V': V_18.tolist(), 'W': W_18.tolist()}
    
    # If that didn't work or time is up, try direct search for 2x3x6 rank 18
    if best_rank > 18 and time.time() < deadline - 10:
        result_direct = _try_rank(18, n, m, p, deadline, seed_offset=200, max_trials=50)
        if result_direct is not None:
            best_rank = 18
            best_result = result_direct
    
    return best_result
# EVOLVE-BLOCK-END
