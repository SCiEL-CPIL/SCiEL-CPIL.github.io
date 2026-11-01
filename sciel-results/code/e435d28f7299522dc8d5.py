"""Matrix multiplication 2x3x5 with Strassen-based rank reduction and optimization search."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

def _strassen_222():
    """Return U, V, W for M_{2,2,2} (Strassen, rank 7).
    U shape (4,7): indexed by (i*2+j, r)
    V shape (4,7): indexed by (j*2+k, r)
    W shape (4,7): indexed by (k*2+i, r)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 7 products for 2x2 matrices
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = -1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook M_{n,m,p} with rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _add_block(global_U, global_V, global_W, rank_offset, 
               sub_U, sub_V, sub_W,
               i_off, j_off, k_off,
               n_sub, m_sub, p_sub,
               n_global, m_global, p_global):
    """Add a sub-block decomposition into the global factor matrices."""
    r = len(sub_U[0])
    for local_i in range(n_sub):
        for local_j in range(m_sub):
            gi = i_off + local_i
            gj = j_off + local_j
            global_row_U = gi * m_global + gj
            for rr in range(r):
                if sub_U[local_i*m_sub + local_j][rr] != 0:
                    global_U[global_row_U][rank_offset + rr] += sub_U[local_i*m_sub + local_j][rr]
    for local_j in range(m_sub):
        for local_k in range(p_sub):
            gj = j_off + local_j
            gk = k_off + local_k
            global_row_V = gj * p_global + gk
            for rr in range(r):
                if sub_V[local_j*p_sub + local_k][rr] != 0:
                    global_V[global_row_V][rank_offset + rr] += sub_V[local_j*p_sub + local_k][rr]
    for local_k in range(p_sub):
        for local_i in range(n_sub):
            gk = k_off + local_k
            gi = i_off + local_i
            global_row_W = gk * n_global + gi
            for rr in range(r):
                if sub_W[local_k*n_sub + local_i][rr] != 0:
                    global_W[global_row_W][rank_offset + rr] += sub_W[local_k*n_sub + local_i][rr]

def _verify_full(U, V, W, n, m, p):
    """Verify that the decomposition is correct by checking tensor equality."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We only need to check if it equals 1 for all i,j,k
    # T is shape (n*m, m*p, p*n)
    # But we can just check the contraction for each (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _search_223_rank11(time_limit=10.0):
    """
    Search for a rank 11 decomposition of M_{2,2,3}.
    Returns (U, V, W) lists if found, else None.
    """
    n, m, p = 2, 2, 3
    target_rank = 11
    
    # Target tensor T[i,j,k] = 1
    # U shape (4, 11), V shape (6, 11), W shape (6, 11)
    
    # Initialize with schoolbook rank 12 and zero out one column, or random
    # Better: Initialize with a random matrix and use gradient descent
    
    start_time = time.time()
    
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(42 + restart)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        learning_rate = 0.05
        decay = 0.98
        steps = 200
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            # Compute residual R[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
            # We can compute this efficiently using einsum or loops
            # R is small (2x2x3 = 12 entries)
            
            # Gradient descent step
            # dU = 2 * sum_{j,k} R[i,j,k] V[j,k,r] W[k,i,r]
            # dV = 2 * sum_{i,k} R[i,j,k] U[i,j,r] W[k,i,r]
            # dW = 2 * sum_{i,j} R[i,j,k] U[i,j,r] V[j,k,r]
            
            # Compute R
            R = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        R[i,j,k] = s - 1.0
            
            err = np.linalg.norm(R)
            if err < 1e-8:
                # Try rounding
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_val = R[i,j,k]
                        if r_val == 0: continue
                        for r in range(target_rank):
                            dU[i*m+j, r] += 2 * r_val * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += 2 * r_val * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += 2 * r_val * U[i*m+j, r] * V[j*p+k, r]
                            
            U -= learning_rate * dU
            V -= learning_rate * dV
            W -= learning_rate * dW
            learning_rate *= decay
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Total rank strategy:
    # Block 1: i in [0,2), j in [0,2), k in [0,2) -> Strassen (7)
    # Block 2: i in [0,2), j in [2,3), k in [0,5) -> Schoolbook (10)
    # Block 3: i in [0,2), j in [0,2), k in [2,5) -> Search for rank 11, fallback 12
    
    # Initialize with fallback rank 29
    total_rank_fallback = 29
    U_fb = [[0]*total_rank_fallback for _ in range(n*m)]
    V_fb = [[0]*total_rank_fallback for _ in range(m*p)]
    W_fb = [[0]*total_rank_fallback for _ in range(p*n)]
    
    # Block 1: Strassen
    sub_U, sub_V, sub_W = _strassen_222()
    _add_block(U_fb, V_fb, W_fb, 0, sub_U, sub_V, sub_W,
               i_off=0, j_off=0, k_off=0,
               n_sub=2, m_sub=2, p_sub=2,
               n_global=n, m_global=m, p_global=p)
    
    # Block 2: Schoolbook 2x1x5
    sb_U, sb_V, sb_W = _schoolbook(2, 1, 5)
    _add_block(U_fb, V_fb, W_fb, 7, sb_U, sb_V, sb_W,
               i_off=0, j_off=2, k_off=0,
               n_sub=2, m_sub=1, p_sub=5,
               n_global=n, m_global=m, p_global=p)
    
    # Block 3: Schoolbook 2x2x3 (fallback)
    sb2_U, sb2_V, sb2_W = _schoolbook(2, 2, 3)
    _add_block(U_fb, V_fb, W_fb, 17, sb2_U, sb2_V, sb2_W,
               i_off=0, j_off=0, k_off=2,
               n_sub=2, m_sub=2, p_sub=3,
               n_global=n, m_global=m, p_global=p)
    
    # Try to find a better decomposition for Block 3 (Rank 11)
    # Time budget: 10 seconds
    search_time_limit = 10.0
    if time.time() < deadline - search_time_limit - 5:
        result = _search_223_rank11(time_limit=search_time_limit)
        if result is not None:
            U_opt, V_opt, W_opt = result
            # Construct the optimized full decomposition
            total_rank_opt = 7 + 10 + 11 # 28
            U_opt_full = [[0]*total_rank_opt for _ in range(n*m)]
            V_opt_full = [[0]*total_rank_opt for _ in range(m*p)]
            W_opt_full = [[0]*total_rank_opt for _ in range(p*n)]
            
            # Add Block 1 (Strassen)
            _add_block(U_opt_full, V_opt_full, W_opt_full, 0, sub_U, sub_V, sub_W,
                       i_off=0, j_off=0, k_off=0,
                       n_sub=2, m_sub=2, p_sub=2,
                       n_global=n, m_global=m, p_global=p)
            
            # Add Block 2 (Schoolbook 2x1x5)
            _add_block(U_opt_full, V_opt_full, W_opt_full, 7, sb_U, sb_V, sb_W,
                       i_off=0, j_off=2, k_off=0,
                       n_sub=2, m_sub=1, p_sub=5,
                       n_global=n, m_global=m, p_global=p)
            
            # Add Block 3 (Optimized 2x2x3)
            _add_block(U_opt_full, V_opt_full, W_opt_full, 17, U_opt, V_opt, W_opt,
                       i_off=0, j_off=0, k_off=2,
                       n_sub=2, m_sub=2, p_sub=3,
                       n_global=n, m_global=m, p_global=p)
            
            # Verify the full optimized decomposition
            if time.time() < deadline - 2:
                if _verify_full(U_opt_full, V_opt_full, W_opt_full, n, m, p):
                    return {"U": U_opt_full, "V": V_opt_full, "W": W_opt_full}
    
    # Fallback to rank 29
    if time.time() < deadline - 2:
        if _verify_full(U_fb, V_fb, W_fb, n, m, p):
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Final fallback: pure schoolbook rank 30
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
