import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def fast_residuals(x, n=2, m=4, p=4, rank=None):
    """Fast residual computation using vectorized operations."""
    if rank is None:
        rank = len(x) // (n*m + m*p + p*n)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U = x[:size_U].reshape(n*m, rank)
    V = x[size_U : size_U + size_V].reshape(m*p, rank)
    W = x[size_U + size_V : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        U_r = U[:, r]
        V_r = V[:, r]
        W_r = W[:, r]
        for i in range(n):
            U_row = U_r[i*m:(i+1)*m]
            for j in range(m):
                V_col = V_r[j*p:(j+1)*p]
                W_col = W_r[i::n]
                T_recon[i, j, :] += U_row[j] * V_col * W_col
    
    return T_recon.flatten() - 1.0

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28) - clean version."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        r = start_r
        
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    return U, V, W

def residuals(x, n=2, m=4, p=4, rank=None):
    """Compute residuals for least_squares - must match the rank in x."""
    if rank is None:
        rank = len(x) // (n*m + m*p + p*n)
    return fast_residuals(x, n, m, p, rank)

def try_rank_reduction(target_rank, n=2, m=4, p=4, max_restarts=100):
    """Try to reduce rank from Strassen's 28 to target_rank using multiple strategies."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    U_s = np.array(strassen_decomposition()[0], dtype=float)
    V_s = np.array(strassen_decomposition()[1], dtype=float)
    W_s = np.array(strassen_decomposition()[2], dtype=float)
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        # Strategy: Start from Strassen truncated, add noise, or random
        if restart == 0:
            U_init = U_s[:, :target_rank].copy()
            V_init = V_s[:, :target_rank].copy()
            W_init = W_s[:, :target_rank].copy()
        elif restart < 20:
            # Small noise on truncation
            noise = 0.1
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * noise
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * noise
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * noise
        elif restart < 40:
            # Medium noise
            noise = 0.5
            U_init = U_s[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * noise
            V_init = V_s[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * noise
            W_init = W_s[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * noise
        else:
            # Random
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use LM method for local optimization
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), 
                                       method='lm', max_nfev=2000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-10:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
            
    # If we got close, try a polish pass
    if best_result is not None and best_error < 0.5:
        try:
            result = opt.least_squares(residuals, best_result, args=(n, m, p, target_rank),
                                       method='lm', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            if err < 1e-10:
                U = result.x[:size_U].reshape(n*m, target_rank)
                V = result.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result.x[size_U + size_V : ].reshape(p*n, target_rank)
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try ranks in order: 27, 26
    ranks_to_try = [27, 26]
    
    for target_r in ranks_to_try:
        if best_rank <= target_r or time_up():
            break
        
        # Check if we have enough time (at least 30s remaining)
        remaining = TIME_LIMIT - (time.time() - START_TIME)
        if remaining < 30:
            break
        
        res = try_rank_reduction(target_r, n, m, p)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = target_r
                best_result = {"U": U2, "V": V2, "W": W2}
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
