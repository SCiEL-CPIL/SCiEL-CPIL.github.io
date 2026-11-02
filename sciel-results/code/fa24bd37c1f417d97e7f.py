"""Combined program: Strassen base + rank reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _strassen_rank50(n, m, p):
    """Deterministic rank-50 Strassen-based decomposition."""
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A_L (2x2) * B_T (2x9) - 4 blocks of 2x2 + 1 block of 2x1
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # Strassen 7 products for 2x2 * 2x2
        U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
        col += 7
    
    # Last column of B_T (k=8)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    W[w_08][col] += 1; W[w_08][col+1] += 1
    W[w_18][col+2] += 1; W[w_18][col+3] += 1
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(mat, tol=1e-4):
    """Round matrix entries to nearest half-integer if close enough."""
    r = np.round(2.0 * mat) / 2.0
    if np.allclose(mat, r, atol=tol):
        return r
    return None

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_time=8.0):
    """Try to reduce from current rank to target_rank using alternating minimization."""
    start = time.time()
    cur_U = np.array(U, dtype=float)
    cur_V = np.array(V, dtype=float)
    cur_W = np.array(W, dtype=float)
    cur_rank = cur_U.shape[1]
    
    if target_rank >= cur_rank:
        return None
    
    # Project to target_rank by keeping first target_rank columns (or best subset)
    # Better: use SVD-like approach - but for tensor, just truncate and refine
    U_t = cur_U[:, :target_rank].copy()
    V_t = cur_V[:, :target_rank].copy()
    W_t = cur_W[:, :target_rank].copy()
    
    # Alternating minimization to fix up the truncated solution
    for iteration in range(100):
        if time.time() - start > max_time - 1:
            break
        
        # Update U: for each (i,j), solve min_u ||M_u u - 1_p||^2
        for i in range(n):
            for j in range(m):
                # M_u[c, k] = V[j*p+k, c] * W[k*n+i, c]
                M_u = np.zeros((target_rank, p))
                for c in range(target_rank):
                    for k in range(p):
                        M_u[c, k] = V_t[j*p+k, c] * W_t[k*n+i, c]
                # Solve (M_u^T M_u) u = M_u^T 1
                A = M_u.T @ M_u + 1e-10 * np.eye(target_rank)
                b = M_u.T @ np.ones(p)
                try:
                    U_t[i*m+j, :] = np.linalg.solve(A, b)
                except:
                    U_t[i*m+j, :] = np.linalg.lstsq(M_u, np.ones(p), rcond=None)[0]
        
        # Update V
        for j in range(m):
            for k in range(p):
                M_v = np.zeros((target_rank, n))
                for c in range(target_rank):
                    for i in range(n):
                        M_v[c, i] = U_t[i*m+j, c] * W_t[k*n+i, c]
                A = M_v.T @ M_v + 1e-10 * np.eye(target_rank)
                b = M_v.T @ np.ones(n)
                try:
                    V_t[j*p+k, :] = np.linalg.solve(A, b)
                except:
                    V_t[j*p+k, :] = np.linalg.lstsq(M_v, np.ones(n), rcond=None)[0]
        
        # Update W
        for k in range(p):
            for i in range(n):
                M_w = np.zeros((target_rank, m))
                for c in range(target_rank):
                    for j in range(m):
                        M_w[c, j] = U_t[i*m+j, c] * V_t[j*p+k, c]
                A = M_w.T @ M_w + 1e-10 * np.eye(target_rank)
                b = M_w.T @ np.ones(m)
                try:
                    W_t[k*n+i, :] = np.linalg.solve(A, b)
                except:
                    W_t[k*n+i, :] = np.linalg.lstsq(M_w, np.ones(m), rcond=None)[0]
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(target_rank):
                        val += U_t[i*m+j, c] * V_t[j*p+k, c] * W_t[k*n+i, c]
                    err += (val - 1.0)**2
        
        if err < 1e-8:
            break
    
    # Check if solution is valid
    U_list = U_t.tolist()
    V_list = V_t.tolist()
    W_list = W_t.tolist()
    if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
        # Try to round to half-integers
        U_r = _round_to_halfint(U_t)
        V_r = _round_to_halfint(V_t)
        W_r = _round_to_halfint(W_t)
        if U_r is not None and V_r is not None and W_r is not None:
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        # Return unrounded if valid
        return U_list, V_list, W_list
    return None

def _random_search(rank, n, m, p, max_time=6.0, n_restarts=10):
    """Random search for rank-r decomposition."""
    start = time.time()
    best_err = float('inf')
    best = None
    
    for restart in range(n_restarts):
        if time.time() - start > max_time - 1:
            break
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for it in range(50):
            if time.time() - start > max_time - 1:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M_u = np.zeros((rank, p))
                    for c in range(rank):
                        for k in range(p):
                            M_u[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    A = M_u.T @ M_u + 1e-10 * np.eye(rank)
                    b = M_u.T @ np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.solve(A, b)
                    except:
                        U[i*m+j, :] = np.linalg.lstsq(M_u, np.ones(p), rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M_v = np.zeros((rank, n))
                    for c in range(rank):
                        for i in range(n):
                            M_v[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    A = M_v.T @ M_v + 1e-10 * np.eye(rank)
                    b = M_v.T @ np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.solve(A, b)
                    except:
                        V[j*p+k, :] = np.linalg.lstsq(M_v, np.ones(n), rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M_w = np.zeros((rank, m))
                    for c in range(rank):
                        for j in range(m):
                            M_w[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    A = M_w.T @ M_w + 1e-10 * np.eye(rank)
                    b = M_w.T @ np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.solve(A, b)
                    except:
                        W[k*n+i, :] = np.linalg.lstsq(M_w, np.ones(m), rcond=None)[0]
            
            # Error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                break
        
        if best_err < 1e-8:
            break
    
    if best is not None and best_err < 1e-6:
        U_t, V_t, W_t = best
        U_list = U_t.tolist()
        V_list = V_t.tolist()
        W_list = W_t.tolist()
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
            U_r = _round_to_halfint(U_t)
            V_r = _round_to_halfint(V_t)
            W_r = _round_to_halfint(W_t)
            if U_r is not None and V_r is not None and W_r is not None:
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
            return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with rank-50 Strassen solution
    U50, V50, W50 = _strassen_rank50(n, m, p)
    
    # Verify it works
    if not _verify(U50, V50, W50, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = 50
    best_result = (U50, V50, W50)
    
    # Try to reduce rank from 49 down
    for target_rank in range(49, 35, -1):
        if time.time() > deadline - 15:
            break
        
        # Method 1: Truncate and refine
        result = _try_reduce_rank(U50, V50, W50, n, m, p, target_rank, max_time=5.0)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-5):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                break
        
        # Method 2: Random search
        if time.time() > deadline - 10:
            break
        result2 = _random_search(target_rank, n, m, p, max_time=4.0, n_restarts=5)
        if result2 is not None:
            U_r, V_r, W_r = result2
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-5):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                break
    
    # If we found something better, return it; otherwise return rank-50
    if best_rank < 50:
        U_r, V_r, W_r = best_result
        return {"U": U_r, "V": V_r, "W": W_r}
    else:
        return {"U": U50, "V": V50, "W": W50}
# EVOLVE-BLOCK-END
