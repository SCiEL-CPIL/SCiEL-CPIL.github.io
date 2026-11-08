import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition as integer lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, T_target):
    """
    Fast vectorized verification.
    """
    T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_calc, T_target, atol=1e-9)

def _construct_tensor(n, m, p):
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    return T_target

def _als_decompose(T, rank, max_iter=1000, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    T is shape (I, J, K).
    Returns U (I, rank), V (J, rank), W (K, rank).
    """
    I, J, K = T.shape
    # Initialize with random values
    U = np.random.randn(I, rank)
    V = np.random.randn(J, rank)
    W = np.random.randn(K, rank)
    
    prev_err = np.inf
    for _ in range(max_iter):
        # Solve for U
        # T is (I, J, K). For each i, T[i, :, :] = sum_r U[i, r] (V[r, :] \otimes W[r, :])
        # Reshape V and W to (J*K, rank)
        VW = np.einsum('jr,kr->jkr', V, W).reshape(J*K, rank)
        # T[i] is (J*K,). We want U[i] such that U[i] @ VW.T approx T[i]
        # Actually, T[i, :, :] reshaped to (J*K,) = sum_r U[i, r] * VW[r, :]
        # So T_flat[i, :] = U[i, :] @ VW.T
        # U = T_flat @ (VW.T @ VW)^-1
        T_flat = T.reshape(I, J*K)
        try:
            U = np.linalg.lstsq(VW, T_flat.T, rcond=None)[0].T
        except:
            pass
            
        # Solve for V
        UW = np.einsum('ir,kr->ikr', U, W).reshape(I*K, rank)
        T_flat = T.reshape(J, I*K) # Careful with axes
        # T[j, :, :] = sum_r V[j, r] * UW[r, :]
        # T_flat[j, :] = V[j, :] @ UW.T
        # Need to transpose T correctly.
        # T is (I, J, K). We want to fix J.
        # T_trans = np.transpose(T, (1, 0, 2)) -> (J, I, K)
        T_trans = np.transpose(T, (1, 0, 2)).reshape(J, I*K)
        try:
            V = np.linalg.lstsq(UW, T_trans.T, rcond=None)[0].T
        except:
            pass
            
        # Solve for W
        UV = np.einsum('ir,jr->ijr', U, V).reshape(I*J, rank)
        # T[k, :, :] = sum_r W[k, r] * UV[r, :]
        # T is (I, J, K). Fix K.
        T_trans = np.transpose(T, (2, 0, 1)).reshape(K, I*J)
        try:
            W = np.linalg.lstsq(UV, T_trans.T, rcond=None)[0].T
        except:
            pass
        
        # Calculate error
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T_calc - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _round_to_half_int(arr):
    # Round to nearest 0.5
    return np.round(arr * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    T_target = _construct_tensor(n, m, p)
    
    # Try ranks from 29 down to 15
    # 30 is the schoolbook rank.
    ranks_to_try = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]
    
    for rank in ranks_to_try:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts for ALS
        max_restarts = 50
        found = False
        for _ in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            try:
                U, V, W = _als_decompose(T_target, rank, max_iter=200, tol=1e-12)
                # Round to half-integers
                U_r = _round_to_half_int(U)
                V_r = _round_to_half_int(V)
                W_r = _round_to_half_int(W)
                
                # Verify
                if _verify(U_r, V_r, W_r, n, m, p, T_target):
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = rank
                    found = True
                    break
            except Exception:
                continue
        
        if found:
            break
            
    return best

# EVOLVE-BLOCK-END
