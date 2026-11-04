import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]
_DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE:
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Default rank 56 solution (Strassen based)
    rank_56 = 56
    U_56 = [[0] * rank_56 for _ in range(n * m)]
    V_56 = [[0] * rank_56 for _ in range(m * p)]
    W_56 = [[0] * rank_56 for _ in range(p * n)]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    def add_block_56(base_rank, i_base, j_base, k_base, U, V, W):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                U[(i_base + li) * 4 + (j_base + lj)][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                V[(j_base + lj) * 8 + (k_base + lk)][cr] += val
            for (lk, li), val in strassen_W[r].items():
                W[(k_base + lk) * 2 + (i_base + li)][cr] += val

    add_block_56(0, 0, 0, 0, U_56, V_56, W_56)
    add_block_56(7, 0, 0, 2, U_56, V_56, W_56)
    add_block_56(14, 0, 0, 4, U_56, V_56, W_56)
    add_block_56(21, 0, 0, 6, U_56, V_56, W_56)
    add_block_56(28, 0, 2, 0, U_56, V_56, W_56)
    add_block_56(35, 0, 2, 2, U_56, V_56, W_56)
    add_block_56(42, 0, 2, 4, U_56, V_56, W_56)
    add_block_56(49, 0, 2, 6, U_56, V_56, W_56)
    
    # Try to find a rank-10 decomposition for 2x2x4
    # using a robust numerical approach with structured initialization
    
    def find_224_rank10(max_time=70):
        """Find rank-10 decomposition for 2x2x4 tensor."""
        start_time = time.time()
        R = 10
        
        # Structured initialization based on Strassen for first 2 cols + 3 extra
        U = np.zeros((2, 2, R))
        V = np.zeros((2, 4, R))
        W = np.zeros((4, 2, R))
        
        # Terms 0-6: Standard Strassen for 2x2x2 (first 2 columns)
        # Term 0: (A00+A11)(B00+B11)
        U[0,0,0] = 1; U[1,1,0] = 1
        V[0,0,0] = 1; V[1,1,0] = 1
        W[0,0,0] = 1; W[1,1,0] = 1
        
        # Term 1: (A10+A11)B00
        U[1,0,1] = 1; U[1,1,1] = 1
        V[0,0,1] = 1
        W[0,1,1] = 1
        
        # Term 2: A00(B01-B11)
        U[0,0,2] = 1
        V[0,1,2] = 1; V[1,1,2] = -1
        W[1,0,2] = 1
        
        # Term 3: A11(B10-B00)
        U[1,1,3] = 1
        V[1,0,3] = 1; V[0,0,3] = -1
        W[0,1,3] = 1; W[0,0,3] = -1
        
        # Term 4: (A00+A10)B11
        U[0,0,4] = 1; U[1,0,4] = 1
        V[1,1,4] = 1
        W[1,1,4] = 1
        
        # Term 5: (A11-A01)(B00+B10)
        U[1,1,5] = 1; U[0,1,5] = -1
        V[0,0,5] = 1; V[1,0,5] = 1
        W[0,0,5] = 1; W[0,1,5] = 1
        
        # Term 6: (A01-A11)(B01+B11)
        U[0,1,6] = 1; U[1,1,6] = -1
        V[0,1,6] = 1; V[1,1,6] = 1
        W[1,0,6] = 1; W[0,0,6] = -1
        
        # Terms 7-9: For columns 2,3
        # Use simple schoolbook for these
        # Term 7: A00*B20
        U[0,0,7] = 1
        V[0,2,7] = 1
        W[2,0,7] = 1
        
        # Term 8: A10*B20
        U[1,0,8] = 1
        V[0,2,8] = 1
        W[2,1,8] = 1
        
        # Term 9: A00*B31 + A10*B31 (combined)
        U[0,0,9] = 1; U[1,0,9] = 1
        V[1,3,9] = 1
        W[3,1,9] = 1; W[3,0,9] = 1
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_l = params[:4*R].reshape(2, 2, R)
            V_l = params[4*R:12*R].reshape(2, 4, R)
            W_l = params[12*R:].reshape(4, 2, R)
            err = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        val = np.dot(U_l[i,j,:], V_l[j,k,:] * W_l[k,i,:])
                        err += (val - 1.0)**2
            return err
        
        # Try optimization from structured init
        try:
            result = minimize(objective, params0, method='L-BFGS-B',
                             options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < 1e-10:
                U_l = result.x[:4*R].reshape(2, 2, R)
                V_l = result.x[4*R:12*R].reshape(2, 4, R)
                W_l = result.x[12*R:].reshape(4, 2, R)
                # Verify
                ok = True
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            val = np.dot(U_l[i,j,:], V_l[j,k,:] * W_l[k,i,:])
                            if abs(val - 1.0) > 1e-8:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    return U_l, V_l, W_l
        except Exception:
            pass
        
        # Try random restarts if structured init failed
        if time.time() - start_time < max_time:
            np.random.seed(42)
            for restart in range(10):
                if time.time() - start_time > max_time:
                    break
                params_rand = np.random.randn(20*R) * 0.3
                try:
                    result = minimize(objective, params_rand, method='L-BFGS-B',
                                     options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-11})
                    if result.fun < 1e-10:
                        U_l = result.x[:4*R].reshape(2, 2, R)
                        V_l = result.x[4*R:12*R].reshape(2, 4, R)
                        W_l = result.x[12*R:].reshape(4, 2, R)
                        ok = True
                        for i in range(2):
                            for j in range(2):
                                for k in range(4):
                                    val = np.dot(U_l[i,j,:], V_l[j,k,:] * W_l[k,i,:])
                                    if abs(val - 1.0) > 1e-8:
                                        ok = False
                                        break
                            if not ok:
                                break
                        if ok:
                            return U_l, V_l, W_l
                except Exception:
                    pass
        
        return None
    
    found_224 = find_224_rank10()
    
    if found_224 is not None:
        U_224, V_224, W_224 = found_224
        R_224 = 10
        
        rank_full = 4 * R_224
        U_full = [[0] * rank_full for _ in range(n * m)]
        V_full = [[0] * rank_full for _ in range(m * p)]
        W_full = [[0] * rank_full for _ in range(p * n)]
        
        def add_block_224(base_rank, i_base, j_base, k_base, U, V, W):
            for r in range(R_224):
                cr = base_rank + r
                for li in range(2):
                    for lj in range(2):
                        U[(i_base + li) * 4 + (j_base + lj)][cr] += U_224[li, lj, r]
                for lj in range(2):
                    for lk in range(4):
                        V[(j_base + lj) * 8 + (k_base + lk)][cr] += V_224[lj, lk, r]
                for lk in range(4):
                    for li in range(2):
                        W[(k_base + lk) * 2 + (i_base + li)][cr] += W_224[lk, li, r]
        
        add_block_224(0, 0, 0, 0, U_full, V_full, W_full)
        add_block_224(R_224, 0, 0, 4, U_full, V_full, W_full)
        add_block_224(2*R_224, 0, 2, 0, U_full, V_full, W_full)
        add_block_224(3*R_224, 0, 2, 4, U_full, V_full, W_full)
        
        try:
            U_arr = np.array(U_full, dtype=float)
            V_arr = np.array(V_full, dtype=float)
            W_arr = np.array(W_full, dtype=float)
            U3 = U_arr.reshape(n, m, rank_full)
            V3 = V_arr.reshape(m, p, rank_full)
            W3 = W_arr.reshape(p, n, rank_full)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            if np.max(np.abs(T - 1.0)) < 1e-8:
                return {"U": U_full, "V": V_full, "W": W_full}
        except Exception:
            pass
    
    return {"U": U_56, "V": V_56, "W": W_56}
# EVOLVE-BLOCK-END
