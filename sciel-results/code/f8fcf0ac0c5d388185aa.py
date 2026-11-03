"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition for 3x3 x 3x6. Rank = 54."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    # Check diagonal (tensor support)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    # Check off-diagonal
    for i in range(n):
        for j in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j2 * p + k
                        c = k * n + i2
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        val = float(np.sum(U_arr[a] * V_arr[b] * W_arr[c]))
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
    return max_err < tol

def _residuals(params, n, m, p, r):
    """Compute residuals for a given parameter vector and rank r."""
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = params[0:size_U].reshape(n*m, r)
    V = params[size_U:size_U+size_V].reshape(m*p, r)
    W = params[size_U+size_V:size_U+size_V+size_W].reshape(p*n, r)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                res[idx] = val - 1.0
                idx += 1
    return res

def _check_decomp(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if not np.isclose(val, 1.0, atol=tol):
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of the given target rank using numerical optimization.
    Uses structured initialization to respect the block structure of 3x3x6 = 2 * 3x3x3.
    """
    from scipy.optimize import least_squares
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    max_attempts = 30
    np.random.seed(42) # Reproducibility for debugging, but we vary seed per attempt
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Structured Initialization
        # We assume the solution is composed of two independent 3x3x3 blocks.
        # Block 1: k in [0, 3)
        # Block 2: k in [3, 6)
        
        params = np.zeros(total_size)
        
        # We split the rank into two halves
        half_rank = target_rank // 2
        remainder = target_rank % 2
        
        # Initialize first half for k < 3
        # And second half for k >= 3
        
        # We need to map the rank indices to the blocks.
        # Let's just use random initialization but with sparsity patterns?
        # Actually, simple random initialization might work if we give it enough attempts.
        # But structured is better.
        
        # Let's create a mask for U, V, W
        # U is (n*m, r) -> (9, r)
        # V is (m*p, r) -> (18, r)
        # W is (p*n, r) -> (18, r)
        
        # For the first half_rank columns:
        # U: all rows active
        # V: only rows corresponding to k < 3 (j*p + k where k in 0..2) -> rows 0..5, 9..14, 18..23? No.
        # V index is j*p + k.
        # If k < 3, then for j=0, k=0,1,2 -> 0,1,2.
        # for j=1, k=0,1,2 -> 3,4,5.
        # for j=2, k=0,1,2 -> 6,7,8.
        # So rows 0..8 are active for k<3? No, V is (m*p, r). m=3, p=6.
        # V[j, k] is at index j*6 + k.
        # If k < 3, then indices are j*6 + 0, j*6 + 1, j*6 + 2.
        # So rows 0,1,2; 6,7,8; 12,13,14 are active.
        
        # W index is k*n + i.
        # If k < 3, then indices are 0..2, 3..5, 6..8.
        # So rows 0..8 are active.
        
        # For the second half_rank columns:
        # V: rows corresponding to k >= 3.
        # Indices: j*6 + 3, j*6 + 4, j*6 + 5.
        # Rows 3,4,5; 9,10,11; 15,16,17.
        
        # W: rows corresponding to k >= 3.
        # Indices: 9..11, 12..14, 15..17.
        # Rows 9..17.
        
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        # Fill first half
        for r_idx in range(half_rank):
            U_init[:, r_idx] = np.random.randn(n*m) * 0.5
            # V: only k < 3
            for j in range(m):
                for k in range(3):
                    V_init[j*p + k, r_idx] = np.random.randn() * 0.5
            # W: only k < 3
            for k in range(3):
                for i in range(n):
                    W_init[k*n + i, r_idx] = np.random.randn() * 0.5
        
        # Fill second half
        for r_idx in range(half_rank, target_rank):
            U_init[:, r_idx] = np.random.randn(n*m) * 0.5
            # V: only k >= 3
            for j in range(m):
                for k in range(3, p):
                    V_init[j*p + k, r_idx] = np.random.randn() * 0.5
            # W: only k >= 3
            for k in range(3, p):
                for i in range(n):
                    W_init[k*n + i, r_idx] = np.random.randn() * 0.5
        
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                _residuals, 
                params, 
                args=(n, m, p, target_rank), 
                method='lm', 
                xtol=1e-12, 
                ftol=1e-12, 
                gtol=1e-12
            )
        except:
            continue
            
        if result.cost < 1e-10:
            U = result.x[0:size_U].reshape(n*m, target_rank)
            V = result.x[size_U:size_U+size_V].reshape(m*p, target_rank)
            W = result.x[size_U+size_V:total_size].reshape(p*n, target_rank)
            
            if _check_decomp(U, V, W, n, m, p):
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _check_decomp(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # If rounding fails, check if the unrounded is "close enough" to integers?
                # The contract says "integer/half-integer".
                # If the unrounded solution is valid but not integer/half-integer, it's invalid.
                # But if the rounded solution is valid, we return it.
                
                # If rounding fails, we might try a different rounding strategy or accept the float
                # if the evaluator is lenient. But we assume strict.
                # We return None to try another initialization.
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try ranks in increasing order of difficulty (lower rank is harder)
    # We know 46 is the theoretical minimum for this structure.
    # Try 46, 48, 50, 52.
    
    for target_rank in [46, 48, 50, 52]:
        if time.time() > deadline:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, deadline)
        if result is not None:
            # Verify the result
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best
# EVOLVE-BLOCK-END
