import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[n*m*p] as a flat list."""
    n, m, p = DIMENSIONS
    size = n * m * p
    T = [0.0] * size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T, n, m, p

def _apply_factors(U, V, W, n, m, p):
    """Apply factor matrices to produce the reconstructed tensor."""
    rank = len(U[0]) if U else 0
    nmp = n * m * p
    result = [0.0] * nmp
    for r in range(rank):
        u_col = [U[i][r] for i in range(n * m)]
        v_col = [V[j][r] for j in range(m * p)]
        w_col = [W[k][r] for k in range(p * n)]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    result[i * m * p + j * p + k] += u_val * v_val * w_val
    return result

def _verify(U, V, W, n, m, p, T):
    """Verify that the decomposition is exact."""
    result = _apply_factors(U, V, W, n, m, p)
    for i in range(len(T)):
        if abs(result[i] - T[i]) > 1e-9:
            return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    n, m, p = 2, 2, 2
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _construct_block_decomposition(n, m, p):
    """
    Construct block decomposition for 2x3x7.
    Rank 39.
    """
    total_rank = 39
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, m_block=1, p=7
    # Schoolbook 2x1x7. Rank 14.
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2, m_block=2, p=7
    # Split p into (0,1), (2,3), (4,5), (6)
    
    sU, sV, sW, srank = _strassen_2x2x2()
    
    # Sub-block 2a: j=1,2; k=0,1. Strassen.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
        
    # Sub-block 2b: j=1,2; k=2,3. Strassen.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
        
    # Sub-block 2c: j=1,2; k=4,5. Strassen.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
        
    # Sub-block 2d: j=1,2; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
            
    return U, V, W, total_rank

def _newton_search(rank, n, m, p, seed, max_iter=50, tol=1e-8):
    """
    Attempt to find a rank-R decomposition using Newton's method.
    Returns (U, V, W) as lists of lists of floats if successful, else None.
    """
    rng = np.random.RandomState(seed)
    
    num_u = n * m * rank
    num_v = m * p * rank
    num_w = p * n * rank
    total_vars = num_u + num_v + num_w
    
    # Initialize with small random values
    x = rng.uniform(-1, 1, total_vars)
    
    def get_tensors(x):
        u_flat = x[:num_u]
        v_flat = x[num_u:num_u+num_v]
        w_flat = x[num_u+num_v:]
        
        U = u_flat.reshape((n*m, rank))
        V = v_flat.reshape((m*p, rank))
        W = w_flat.reshape((p*n, rank))
        return U, V, W
    
    def compute_residuals(x):
        U, V, W = get_tensors(x)
        residuals = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residuals[idx] = 1.0 - s
                    idx += 1
        return residuals
    
    def compute_jacobian(x):
        U, V, W = get_tensors(x)
        J = np.zeros((n*m*p, total_vars))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                    for r in range(rank):
                        var_idx_u = (i*m+j)*rank + r
                        J[idx, var_idx_u] = V[j*p+k, r] * W[k*n+i, r]
                        
                        var_idx_v = num_u + (j*p+k)*rank + r
                        J[idx, var_idx_v] = U[i*m+j, r] * W[k*n+i, r]
                        
                        var_idx_w = num_u + num_v + (k*n+i)*rank + r
                        J[idx, var_idx_w] = U[i*m+j, r] * V[j*p+k, r]
                    idx += 1
        return J
    
    for _ in range(max_iter):
        res = compute_residuals(x)
        if np.linalg.norm(res) < tol:
            break
        
        J = compute_jacobian(x)
        try:
            # Use lstsq for robustness
            delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        except Exception:
            break
            
        x = x + delta
        
    res = compute_residuals(x)
    if np.linalg.norm(res) < 1e-6:
        U, V, W = get_tensors(x)
        U_list = [[float(v) for v in row] for row in U]
        V_list = [[float(v) for v in row] for row in V]
        W_list = [[float(v) for v in row] for row in W]
        return U_list, V_list, W_list
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    T, n, m, p = _build_tensor()
    
    # Fallback: Block decomposition rank 39
    U_block, V_block, W_block, rank_block = _construct_block_decomposition(n, m, p)
    best_U, best_V, best_W = U_block, V_block, W_block
    best_rank = rank_block
    
    # Try to find lower ranks using Newton's method
    # Lower bound is max(n*m, m*p, p*n) = 21
    # We know 39 works. Let's try 30 to 38.
    # We prioritize lower ranks.
    
    for rank in range(30, 39):
        if time.time() > deadline - 10: # Leave time for verification and return
            break
            
        found = False
        # Try a few random seeds
        for seed in range(20):
            if time.time() > deadline - 10:
                break
                
            result = _newton_search(rank, n, m, p, seed)
            if result:
                U, V, W = result
                if _verify(U, V, W, n, m, p, T):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break # Found valid for this rank, try next lower rank
        
    return {"U": best_U, "V": best_V, "W": best_W}

