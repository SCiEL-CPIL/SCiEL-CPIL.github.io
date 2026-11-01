import time
import numpy as np

_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Reconstruct T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Using einsum for speed
    # U is (n*m, rank) -> reshape to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, T_recon, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """Construct the rank-14 decomposition using two independent Strassen blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    # Indices:
    # U: i*m + j. A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V: j*p + k. B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # W: k*n + i. C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    
    def add_strassen_block_offset(rank_offset, b_idx_map, c_idx_map):
        # m1
        r = rank_offset + 0
        U[0, r] = 1; U[3, r] = 1
        V[b_idx_map[0], r] = 1; V[b_idx_map[3], r] = 1
        W[c_idx_map[0], r] = 1; W[c_idx_map[3], r] = 1
        
        # m2
        r = rank_offset + 1
        U[2, r] = 1; U[3, r] = 1
        V[b_idx_map[0], r] = 1
        W[c_idx_map[2], r] = 1; W[c_idx_map[3], r] = -1
        
        # m3
        r = rank_offset + 2
        U[0, r] = 1
        V[b_idx_map[2], r] = 1; V[b_idx_map[3], r] = -1
        W[c_idx_map[1], r] = 1; W[c_idx_map[3], r] = 1
        
        # m4
        r = rank_offset + 3
        U[3, r] = 1
        V[b_idx_map[1], r] = 1; V[b_idx_map[0], r] = -1
        W[c_idx_map[0], r] = 1; W[c_idx_map[2], r] = 1
        
        # m5
        r = rank_offset + 4
        U[0, r] = 1; U[1, r] = 1
        V[b_idx_map[3], r] = 1
        W[c_idx_map[1], r] = 1; W[c_idx_map[0], r] = -1
        
        # m6
        r = rank_offset + 5
        U[2, r] = 1; U[0, r] = -1
        V[b_idx_map[0], r] = 1; V[b_idx_map[1], r] = 1
        W[c_idx_map[3], r] = 1
        
        # m7
        r = rank_offset + 6
        U[1, r] = 1; U[3, r] = -1
        V[b_idx_map[2], r] = 1; V[b_idx_map[3], r] = 1
        W[c_idx_map[0], r] = 1

    # Left Half
    b_left = {0: 0, 1: 1, 2: 4, 3: 5}
    c_left = {0: 0, 1: 1, 2: 2, 3: 3}
    add_strassen_block_offset(0, b_left, c_left)
    
    # Right Half
    b_right = {0: 2, 1: 3, 2: 6, 3: 7}
    c_right = {0: 4, 1: 5, 2: 6, 3: 7}
    add_strassen_block_offset(7, b_right, c_right)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank13(time_budget=10.0):
    """
    Attempt to find a rank-13 decomposition using randomized Alternating Least Squares (ALS).
    Returns None if not found within time budget.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    
    # Tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    start_time = time.time()
    remaining = min(time_budget, _DEADLINE - (time.time() - start_time) + start_time)
    remaining = min(time_budget, 140.0 - 5.0)  # safety margin
    
    for restart in range(300):
        if time.time() - start_time > remaining - 3.0:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # ALS iterations
        for it in range(80):
            if time.time() - start_time > remaining - 3.0:
                break
            
            # Solve for W: for each (k,i), solve M @ w = ones(m)
            # M[j,r] = U[i*m+j,r] * V[j*p+k,r]
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
            
            # Solve for U: for each (i,j), solve M @ u = ones(p)
            # M[k,r] = V[j*p+k,r] * W[k*n+i,r]
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
            
            # Solve for V: for each (j,k), solve M @ v = ones(n)
            # M[i,r] = U[i*m+j,r] * W[k*n+i,r]
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
        
        # Check convergence
        U_reshaped = U.reshape(n, m, target_rank)
        V_reshaped = V.reshape(m, p, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        error = np.max(np.abs(T - T_recon))
        
        if error < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _round_to_half_int(A):
    """Round a numpy array to the nearest half-integer."""
    return np.round(A * 2) / 2.0

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    
    # Attempt to find a lower rank (13) with a strict time budget
    # Reserve time for verification and fallback
    search_budget = min(10.0, 140.0 - (time.time() - start_time) - 5.0)
    if search_budget < 1.0:
        search_budget = 1.0
    
    result_13 = _search_rank13(time_budget=search_budget)
    
    if result_13 is not None:
        if time.time() - start_time < 140.0 - 3.0:
            # Convert to numpy arrays
            U_arr = np.array(result_13["U"], dtype=float)
            V_arr = np.array(result_13["V"], dtype=float)
            W_arr = np.array(result_13["W"], dtype=float)
            
            # Round to nearest half-integers
            U_r = _round_to_half_int(U_arr)
            V_r = _round_to_half_int(V_arr)
            W_r = _round_to_half_int(W_arr)
            
            # Verify the rounded solution exactly
            if _verify(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to the known rank-14 solution
    res = _strassen_rank14()
    
    if _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
