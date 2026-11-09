import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    error = np.linalg.norm(T - 1.0)
    return error

def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _try_rank(R, n, m, p, time_limit=10, num_restarts=5):
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
            U_mat = U_flat.reshape(n, m, R)
            V_mat = V_flat.reshape(m, p, R)
            W_mat = W_flat.reshape(p, n, R)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
            err = np.sum((T - 1.0)**2)
            return err
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            x_opt = res.x
            U_flat, V_flat, W_flat = x_opt[:n*m*R], x_opt[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
            U_opt = U_flat.reshape(n*m, R)
            V_opt = V_flat.reshape(m*p, R)
            W_opt = W_flat.reshape(p*n, R)
            
            # Snap and verify
            U_snap = _snap_to_half(U_opt)
            V_snap = _snap_to_half(V_opt)
            W_snap = _snap_to_half(W_opt)
            
            err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
            
            if err_snap < 1e-6:
                return U_snap, V_snap, W_snap
                
            if err_snap < best_error:
                best_error = err_snap
                best_factors = (U_snap, V_snap, W_snap)
                
        except Exception:
            pass
            
    return best_factors if best_error < 1e-6 else None

def _construct_rank_17():
    """
    Construct a Rank 17 decomposition for M_{2,3,3} using
    Rank 11 for M_{2,3,2} (isomorphic to M_{2,2,3}) and Rank 6 for M_{2,3,1}.
    """
    n, m, p = 2, 3, 3
    R = 17
    
    # 1. Get Rank 11 for M_{2,2,3}
    # M_{2,2,3}: n=2, m=2, p=3
    sub_res = _try_rank(11, 2, 2, 3, time_limit=15, num_restarts=10)
    
    if sub_res is None:
        return None
        
    U_sub, V_sub, W_sub = sub_res
    # U_sub: (4, 11) for A(2x2)
    # V_sub: (6, 11) for B(2x3)
    # W_sub: (6, 11) for C(3x2)
    
    # Map M_{2,2,3} to M_{2,3,2}
    # M_{2,2,3}: i in {0,1}, j in {0,1}, k in {0,1,2}
    # M_{2,3,2}: i in {0,1}, j in {0,1,2}, k in {0,1}
    # Isomorphism: i' = i, j' = k, k' = j
    # A'_{i'j'} = B_{j'i'}? No.
    # The tensor T_{ijk} for M_{2,2,3} is A_{ij} B_{jk} C_{ki}.
    # The tensor T'_{i'j'k'} for M_{2,3,2} is A'_{i'j'} B'_{j'k'} C'_{k'i'}.
    # We want T' to match T under permutation of indices.
    # Let's just construct the full M_{2,3,2} decomposition directly if possible?
    # No, we have the factors for M_{2,2,3}.
    
    # Let's build the factors for M_{2,3,2} (n=2, m=3, p=2)
    # U_1: (6, 11)
    # V_1: (6, 11)
    # W_1: (4, 11)
    
    U_1 = np.zeros((6, 11))
    V_1 = np.zeros((6, 11))
    W_1 = np.zeros((4, 11))
    
    # Mapping indices
    # Sub: i in 0..1, j in 0..1, k in 0..2
    # Main: i in 0..1, j in 0..2, k in 0..1
    # We identify i_main = i_sub, k_main = j_sub, j_main = k_sub
    # So for each r in 0..10:
    # U_sub[i_sub, j_sub, r] -> U_1[i_main, j_main, r] = U_sub[i_main, k_main, r]
    # V_sub[j_sub, k_sub, r] -> V_1[j_main, k_main, r] = V_sub[k_main, j_main, r]
    # W_sub[k_sub, i_sub, r] -> W_1[k_main, i_main, r] = W_sub[j_main, i_main, r]
    
    for r in range(11):
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # U_sub index: i * 2 + j
                    # U_1 index: i * 3 + k
                    U_1[i * 3 + k, r] = U_sub[i * 2 + j, r]
                    
                    # V_sub index: j * 3 + k
                    # V_1 index: k * 2 + j
                    V_1[k * 2 + j, r] = V_sub[j * 3 + k, r]
                    
                    # W_sub index: k * 2 + i
                    # W_1 index: j * 2 + i
                    W_1[j * 2 + i, r] = W_sub[k * 2 + i, r]
                    
    # Verify M_{2,3,2} decomposition
    # err = _verify(U_1, V_1, W_1, 2, 3, 2)
    # if err > 1e-6: return None
    
    # 2. Construct Rank 6 for M_{2,3,1}
    # M_{2,3,1}: n=2, m=3, p=1
    # U_2: (6, 6)
    # V_2: (3, 6)
    # W_2: (2, 6)
    
    U_2 = np.zeros((6, 6))
    V_2 = np.zeros((3, 6))
    W_2 = np.zeros((2, 6))
    
    col = 0
    for i in range(2):
        for j in range(3):
            U_2[i * 3 + j, col] = 1
            V_2[j, col] = 1
            W_2[i, col] = 1
            col += 1
            
    # 3. Combine
    U = np.hstack([U_1, U_2])
    V = np.hstack([V_1, V_2])
    W = np.hstack([W_1, W_2])
    
    # Verify full M_{2,3,3}
    err = _verify(U, V, W, 2, 3, 3)
    if err < 1e-6:
        return U, V, W
        
    return None

def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    deadline = global_start + 130 # Safety margin
    
    # 1. Try Rank 15
    if time.time() < deadline - 40:
        res = _try_rank(15, n, m, p, time_limit=20, num_restarts=5)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Try Rank 16
    if time.time() < deadline - 30:
        res = _try_rank(16, n, m, p, time_limit=15, num_restarts=5)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3. Construct Rank 17
    if time.time() < deadline - 20:
        res = _construct_rank_17()
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 4. Try Rank 17 optimization (in case construction failed)
    if time.time() < deadline - 10:
        res = _try_rank(17, n, m, p, time_limit=10, num_restarts=5)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
