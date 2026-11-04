"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x6 matrix.
    Strategy: Find a low-rank decomposition for 3x3x3 (target 23),
    then extend to 3x3x6 by duplicating the U factors and splitting V, W.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook fallback for 3x3x6
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    
    def validate(U, V, W, T):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < 1e-9):
                return False
        return True

    def to_int_or_half(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return np.round(arr * 2) / 2.0

    def optimize_333(U_init, V_init, W_init, T, max_iters=10000, lr_init=0.01):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        mU, vU = np.zeros_like(U), np.zeros_like(U)
        mV, vV = np.zeros_like(V), np.zeros_like(V)
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = T_rec - T
            err_sq = np.sum(err**2)
            
            if err_sq < best_err:
                best_err = err_sq
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err_sq < 1e-12:
                break
            
            gU = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
            gV = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
            gW = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
            
            t += 1
            lr = lr_init * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            mU = beta1 * mU + (1 - beta1) * gU
            vU = beta2 * vU + (1 - beta2) * gU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * gV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        return best_U, best_V, best_W, best_err

    def find_333_decomp(target_rank):
        """Find a rank-target_rank decomposition for 3x3x3."""
        n3 = 3
        T333 = np.zeros((n3*n3, n3*n3, n3*n3), dtype=np.float64)
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    T333[i*n3+j, j*n3+k, k*n3+i] = 1.0
        
        best_result = None
        # Try a few restarts
        n_restarts = 10
        for seed in range(n_restarts):
            if time.time() > deadline - 20:
                break
            np.random.seed(seed * 1000 + target_rank)
            U_init = np.random.randn(n3*n3, target_rank) * 0.5
            V_init = np.random.randn(n3*n3, target_rank) * 0.5
            W_init = np.random.randn(n3*n3, target_rank) * 0.5
            
            U_opt, V_opt, W_opt, err_sq = optimize_333(U_init, V_init, W_opt, T333, max_iters=5000)
            
            if err_sq < 1e-10:
                U_r = to_int_or_half(U_opt)
                V_r = to_int_or_half(V_opt)
                W_r = to_int_or_half(W_opt)
                
                if validate(U_r, V_r, W_r, T333):
                    best_result = (U_r, V_r, W_r)
                    break
        return best_result

    # Try to find a low-rank 3x3 decomposition
    # Ranks to try: 23, 24, 25, 26
    # If found, extend to 3x3x6
    best_333 = None
    best_rank_333 = 27 # schoolbook
    
    for r in [23, 24, 25, 26]:
        if time.time() > deadline - 30:
            break
        res = find_333_decomp(r)
        if res is not None:
            best_333 = res
            best_rank_333 = r
            break # Found the lowest possible (or at least a good one)
            
    if best_333 is not None:
        U333, V333, W333 = best_333
        r333 = best_rank_333
        total_rank = 2 * r333
        
        U_total = np.zeros((n*m, total_rank), dtype=np.float64)
        V_total = np.zeros((m*p, total_rank), dtype=np.float64)
        W_total = np.zeros((p*n, total_rank), dtype=np.float64)
        
        # First block (cols 0-2)
        U_total[:, :r333] = U333
        for j in range(n):
            for k in range(3):
                V_total[j*p+k, :r333] = V333[j*3+k, :r333]
        for k in range(3):
            for i in range(n):
                W_total[k*n+i, :r333] = W333[k*3+i, :r333]
                
        # Second block (cols 3-5)
        U_total[:, r333:] = U333
        for j in range(n):
            for k in range(3):
                # Map k (0,1,2) to p indices (3,4,5)
                V_total[j*p+(k+3), r333:] = V333[j*3+k, :r333]
        for k in range(3):
            for i in range(n):
                # Map k (0,1,2) to p indices (3,4,5)
                W_total[(k+3)*n+i, r333:] = W333[k*3+i, :r333]
                
        # Validate the full 3x3x6 decomposition
        T_full = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i*m+j, j*p+k, k*n+i] = 1.0
                    
        if validate(U_total, V_total, W_total, T_full):
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
        else:
            # If validation fails (should not happen), fallback
            pass

    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
