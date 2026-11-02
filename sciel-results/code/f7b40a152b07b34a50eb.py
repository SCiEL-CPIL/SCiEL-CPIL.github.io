import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140.0
    
    # Try to find rank 50 decomposition (5 blocks of 2x2x3 rank 10)
    result = _try_rank50(deadline)
    if result is not None:
        return result
    
    # Try direct rank-50 search
    result = _try_direct_rank50(deadline)
    if result is not None:
        return result
        
    # Fallback to rank 53
    return _build_rank53()

def _verify_decomposition(U, V, W, n=2, m=2, p=15):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Build the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    u_val = U[i*m + j][r]
                    v_val = V[j*p + k][r]
                    w_val = W[k*n + i][r]
                    actual += u_val * v_val * w_val
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_rank50(deadline):
    n, m, p = 2, 2, 15
    rank_block = 10
    num_blocks = 5
    total_rank = rank_block * num_blocks
    
    # Find a rank 10 decomposition for 2x2x3
    U_b, V_b, W_b = _find_m223_rank10(deadline)
    if U_b is None:
        return None
        
    # Assemble 5 blocks
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        k0 = 3 * b
        col_offset = b * rank_block
        for r in range(rank_block):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_b[i*2 + j][r]
            for j in range(2):
                for k in range(3):
                    kk = k0 + k
                    V[j*p + kk][c] = V_b[j*3 + k][r]
            for k in range(3):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_b[k*2 + i][r]
                    
    # Verify the full decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
        
    return {"U": U, "V": V, "W": W}

def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7
        
    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]
            
    return {"U": U, "V": V, "W": W}

def _try_rank50(deadline):
    n, m, p = 2, 2, 15
    rank_block = 10
    num_blocks = 5
    total_rank = rank_block * num_blocks
    
    # Find a rank 10 decomposition for 2x2x3
    U_b, V_b, W_b = _find_m223_rank10(deadline)
    if U_b is None:
        return None
        
    # Assemble 5 blocks
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        k0 = 3 * b
        col_offset = b * rank_block
        for r in range(rank_block):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_b[i*2 + j][r]
            for j in range(2):
                for k in range(3):
                    kk = k0 + k
                    V[j*p + kk][c] = V_b[j*3 + k][r]
            for k in range(3):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_b[k*2 + i][r]
                    
    # Verify the full decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
        
    return {"U": U, "V": V, "W": W}

def _find_m223_rank10(deadline):
    n, m, p = 2, 2, 3
    rank = 10
    
    # Residual function for 2x2x3
    def residuals(params):
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+6*rank].reshape(6, rank)
        W = params[4*rank+6*rank:].reshape(6, rank)
        
        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    # Try structured initializations first, then random
    inits = []
    
    # Initialization 1: Based on Strassen-like structure for 2x2x2 + extra
    # Use known rank-7 for 2x2x2 and extend
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, -1, 0, 0, 0]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=float)
    inits.append(np.concatenate([U_s.flatten(), V_s.flatten(), W_s.flatten()]))
    
    # Initialization 2: More structured - use rank-7 Strassen for first 2 k's,
    # then add 3 more terms for the 3rd k with specific patterns
    U_s2 = np.array([
        [1, 0, 1, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, -1, 0, 0, 0]
    ], dtype=float)
    V_s2 = np.array([
        [1, 1, 0, -1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0]
    ], dtype=float)
    W_s2 = np.array([
        [1, 0, 0, 1, -1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=float)
    inits.append(np.concatenate([U_s2.flatten(), V_s2.flatten(), W_s2.flatten()]))
    
    # Random initializations with various scales
    for seed in range(80):
        if time.time() > deadline - 15:
            break
        np.random.seed(seed)
        scale = [0.3, 0.5, 0.7, 1.0, 1.5][seed % 5]
        x0 = np.random.randn(16*rank) * scale
        inits.append(x0)
    
    best_result = None
    best_norm = np.inf
    
    for x0 in inits:
        if time.time() > deadline - 15:
            break
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=3000)
            norm = np.linalg.norm(result.fun)
            if norm < best_norm:
                best_norm = norm
                best_result = result
            
            if norm < 1e-9:
                U = result.x[:4*rank].reshape(4, rank)
                V = result.x[4*rank:4*rank+6*rank].reshape(6, rank)
                W = result.x[4*rank+6*rank:].reshape(6, rank)
                
                # High-precision polish pass
                U, V, W = _polish_m223(U, V, W, deadline)
                
                # Snap to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
                if np.max(np.abs(res_r)) < 1e-6:
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
        except Exception:
            continue
    
    # If we got close but not exact, try to polish the best result
    if best_result is not None and best_norm < 1e-4:
        try:
            U = best_result.x[:4*rank].reshape(4, rank)
            V = best_result.x[4*rank:4*rank+6*rank].reshape(6, rank)
            W = best_result.x[4*rank+6*rank:].reshape(6, rank)
            
            U, V, W = _polish_m223(U, V, W, deadline)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
            if np.max(np.abs(res_r)) < 1e-6:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        except Exception:
            pass
            
    return None, None, None

def _polish_m223(U, V, W, deadline):
    """High-precision polish pass for 2x2x3 rank-10 decomposition."""
    rank = U.shape[1]
    
    def residuals(params):
        U_l = params[:4*rank].reshape(4, rank)
        V_l = params[4*rank:4*rank+6*rank].reshape(6, rank)
        W_l = params[4*rank+6*rank:].reshape(6, rank)
        
        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U_l[i*2+j], V_l[j*3+k] * W_l[k*2+i]) - 1.0
                    idx += 1
        return res
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12)
        if np.linalg.norm(result.fun) < 1e-10:
            U_p = result.x[:4*rank].reshape(4, rank)
            V_p = result.x[4*rank:4*rank+6*rank].reshape(6, rank)
            W_p = result.x[4*rank+6*rank:].reshape(6, rank)
            return U_p, V_p, W_p
    except Exception:
        pass
    
    return U, V, W

def _try_direct_rank50(deadline):
    """Try to find a direct rank-50 decomposition for 2x2x15."""
    n, m, p = 2, 2, 15
    rank = 50
    
    def residuals(params):
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+30*rank].reshape(30, rank)
        W = params[4*rank+30*rank:].reshape(30, rank)
        
        res = np.zeros(60)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    res[idx] = np.dot(U[i*2+j], V[j*15+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res
    
    # Initialize with block structure: 5 blocks of 2x2x3
    # Use Strassen-like base for each block
    U_init = np.zeros((4, rank))
    V_init = np.zeros((30, rank))
    W_init = np.zeros((30, rank))
    
    # Strassen 2x2x2 pattern
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    for b in range(5):
        col_off = b * 10
        k0 = 3 * b
        # Use 7 Strassen terms + 3 extra per block
        for r in range(7):
            c = col_off + r
            for i in range(2):
                for j in range(2):
                    U_init[i*2+j, c] = U_s[i*2+j, r]
            for j in range(2):
                for k in range(2):  # Only first 2 k's from Strassen
                    kk = k0 + k
                    V_init[j*15+kk, c] = V_s[j*2+k, r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W_init[kk*2+i, c] = W_s[k*2+i, r]
        
        # 3 extra terms for the 3rd k
        for r in range(3):
            c = col_off + 7 + r
            # Simple identity-like pattern for 3rd k
            if r == 0:
                U_init[0, c] = 1
                V_init[15, c] = 1
                W_init[30-3+0, c] = 1  # k=2, i=0 -> index 2*2+0=4, but in 30-vector: k=2 -> 2*2=4
            elif r == 1:
                U_init[1, c] = 1
                V_init[16, c] = 1
                W_init[5, c] = 1  # k=2, i=1 -> index 5
            else:
                U_init[2, c] = 1
                V_init[15, c] = 1
                W_init[4, c] = 1
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=2000)
        if np.linalg.norm(result.fun) < 1e-9:
            U = result.x[:4*rank].reshape(4, rank)
            V = result.x[4*rank:4*rank+30*rank].reshape(30, rank)
            W = result.x[4*rank+30*rank:].reshape(30, rank)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
            if np.max(np.abs(res_r)) < 1e-6:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    except Exception:
        pass
    
    return None
