"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 multiplication."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _schoolbook_221():
    """Return schoolbook decomposition for 2x2x1 multiplication."""
    U = np.eye(4)
    V = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]], dtype=float)
    W = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1]], dtype=float)
    return U, V, W

def _residuals_m22p(params, p, rank):
    """Compute residuals for M_{2,2,p} decomposition."""
    n, m = 2, 2
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
    W = params[(n*m*rank + m*p*rank):].reshape(p*n, rank)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                idx += 1
    return res

def _find_decomp_m22p(p, target_rank, deadline, max_seeds=20):
    """Search for rank-target_rank decomposition of M_{2,2,p}."""
    n, m = 2, 2
    total_vars = (n*m + m*p + p*n) * target_rank
    
    def objective(params):
        res = _residuals_m22p(params, p, target_rank)
        return np.sum(res**2)
    
    best_params = None
    best_obj = np.inf
    
    for seed in range(max_seeds):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        x0 = np.random.randn(total_vars) * 0.5
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
            if best_obj < 1e-10:
                break
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    U = best_params[:n*m*target_rank].reshape(n*m, target_rank)
    V = best_params[n*m*target_rank:(n*m*target_rank + m*p*target_rank)].reshape(m*p, target_rank)
    W = best_params[(n*m*target_rank + m*p*target_rank):].reshape(p*n, target_rank)
    
    # Snap to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify
    res = _residuals_m22p(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), p, target_rank)
    if np.max(np.abs(res)) < 1e-6:
        return U, V, W
    return None

def _build_rank53():
    """Build the guaranteed rank-53 decomposition (7 Strassen blocks + 1 schoolbook)."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen, V_strassen, W_strassen = _strassen_222()
    U_school, V_school, W_school = _schoolbook_221()
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = int(U_strassen[i * 2 + j, r])
            for j in range(2):
                V[j * p + k0][c] = int(V_strassen[j * 2, r])
                V[j * p + k1][c] = int(V_strassen[j * 2 + 1, r])
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = int(W_strassen[k * 2 + i, r])
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = int(U_school[i * 2 + j, r])
        for j in range(2):
            V[j * p + k_last][c] = int(V_school[j, r])
        for i in range(2):
            W[k_last * n + i][c] = int(W_school[i, r])
    
    return {"U": U, "V": V, "W": W}

def _build_block_decomp(U_block, V_block, W_block, p_block, k_start, U, V, W, col_offset, rank):
    """Fill in a block decomposition starting at column col_offset and row k_start."""
    n, m = 2, 2
    for r in range(p_block):
        c = col_offset + r
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = float(U_block[i * m + j, r])
        for j in range(m):
            for k in range(p_block):
                V[j * p_block + k][c] = float(V_block[j * p_block + k, r])
        for k in range(p_block):
            for i in range(n):
                W[k * n + i][c] = float(W_block[k * n + i, r])

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Fallback: rank-53 solution
    fallback = _build_rank53()
    
    # Strategy: decompose 2x15 into 2x9 + 2x6
    # Try to find lower-rank decompositions for each block
    
    # First, try to find a rank-10 decomposition for M_{2,2,9}
    # (3 blocks of M_{2,2,3}, each with rank 10, but we search for better)
    # Actually, let's try rank 9 for M_{2,2,9} - this would be an improvement
    
    # Try M_{2,2,9} with rank 10 first (known achievable via 3x M_{2,2,3})
    m229 = _find_decomp_m22p(9, 10, deadline, max_seeds=8)
    
    if m229 is not None and time.time() < deadline:
        U9, V9, W9 = m229
        # Now try M_{2,2,6} with rank 15 (3x M_{2,2,2} = 21, but try 15)
        m226 = _find_decomp_m22p(6, 15, deadline, max_seeds=10)
        
        if m226 is not None:
            U6, V6, W6 = m226
            total_rank = 10 + 15  # = 25
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Fill 2x9 block (k=0..8)
            for r in range(10):
                c = r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][c] = float(U9[i * m + j, r])
                for j in range(m):
                    for k in range(9):
                        V[j * p + k][c] = float(V9[j * 9 + k, r])
                for k in range(9):
                    for i in range(n):
                        W[k * n + i][c] = float(W9[k * 2 + i, r])
            
            # Fill 2x6 block (k=9..14)
            for r in range(15):
                c = 10 + r
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][c] = float(U6[i * m + j, r])
                for j in range(m):
                    for k in range(6):
                        V[j * p + 9 + k][c] = float(V6[j * 6 + k, r])
                for k in range(6):
                    for i in range(n):
                        W[(9 + k) * n + i][c] = float(W6[k * 2 + i, r])
            
            return {"U": U, "V": V, "W": W}
    
    # Try alternative: 2x5 + 2x5 + 2x5 with rank 12 each = 36 total
    if time.time() < deadline:
        m225 = _find_decomp_m22p(5, 12, deadline, max_seeds=10)
        if m225 is not None:
            U5, V5, W5 = m225
            total_rank = 36
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            for block in range(3):
                k_start = 5 * block
                col_offset = 12 * block
                for r in range(12):
                    c = col_offset + r
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][c] = float(U5[i * m + j, r])
                    for j in range(m):
                        for k in range(5):
                            V[j * p + k_start + k][c] = float(V5[j * 5 + k, r])
                    for k in range(5):
                        for i in range(n):
                            W[(k_start + k) * n + i][c] = float(W5[k * 2 + i, r])
            
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-53
    return fallback
# EVOLVE-BLOCK-END
