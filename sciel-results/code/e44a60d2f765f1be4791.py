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
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    T_target: (n*m, m*p, p*n)
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

def _als_decompose(T, rank, max_iter=500, tol=1e-12, init=None):
    """
    Alternating Least Squares for tensor decomposition.
    T is shape (I, J, K).
    Returns U (I, rank), V (J, rank), W (K, rank).
    """
    I, J, K = T.shape
    
    if init is not None:
        U, V, W = init
    else:
        U = np.random.randn(I, rank)
        V = np.random.randn(J, rank)
        W = np.random.randn(K, rank)
        
    # Normalize initial factors to prevent divergence
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)

    prev_err = np.inf
    
    for _ in range(max_iter):
        # 1. Solve for U
        # T[i, :, :] = sum_r U[i, r] (V[r, :] \otimes W[r, :])
        # Flatten V and W to (J*K, rank)
        VW = (V[:, :, None] * W[None, :, :]).reshape(J*K, rank)
        # T_flat[i, :] is T[i, :, :] flattened
        T_flat = T.reshape(I, J*K)
        # We want U[i, :] such that U[i, :] @ VW.T approx T_flat[i, :]
        # This is a linear system: VW @ U[i, :].T = T_flat[i, :].T
        try:
            U = np.linalg.lstsq(VW, T_flat.T, rcond=None)[0].T
        except Exception:
            pass
            
        # 2. Solve for V
        # T[:, j, :] = sum_r V[j, r] (U[r, :] \otimes W[r, :])
        UW = (U[:, :, None] * W[None, :, :]).reshape(I*K, rank)
        # T_trans[j, :] is T[:, j, :] flattened. Note: T is (I, J, K).
        # We need to move axis 1 to 0, then flatten the rest.
        T_trans = np.transpose(T, (1, 0, 2)).reshape(J, I*K)
        try:
            V = np.linalg.lstsq(UW, T_trans.T, rcond=None)[0].T
        except Exception:
            pass
            
        # 3. Solve for W
        # T[:, :, k] = sum_r W[k, r] (U[r, :] \otimes V[r, :])
        UV = (U[:, :, None] * V[None, :, :]).reshape(I*J, rank)
        # T_trans2[k, :] is T[:, :, k] flattened. Move axis 2 to 0.
        T_trans2 = np.transpose(T, (2, 0, 1)).reshape(K, I*J)
        try:
            W = np.linalg.lstsq(UV, T_trans2.T, rcond=None)[0].T
        except Exception:
            pass
        
        # Calculate error
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T_calc - T)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _round_to_half_int(arr):
    return np.round(arr * 2) / 2

def _local_search_rounding(U, V, W, T_target, n, m, p, max_steps=100):
    """
    Attempts to refine the rounded solution by flipping values to nearest neighbors 
    to reduce error to zero.
    """
    U_r = _round_to_half_int(U)
    V_r = _round_to_half_int(V)
    W_r = _round_to_half_int(W)
    
    current_err = np.linalg.norm(np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r) - T_target)
    
    if current_err < 1e-9:
        return U_r, V_r, W_r
    
    # If error is small, try local perturbations
    # This is a heuristic: if we are very close, maybe a single flip fixes it?
    # Given the complexity, we will just return the rounded version if it's not perfect,
    # but we can try a few random flips if error is extremely small.
    
    if current_err < 1e-6:
        for _ in range(max_steps):
            # Pick a random element from one of the matrices
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                i, j = np.random.randint(U_r.shape[0]), np.random.randint(U_r.shape[1])
                orig = U_r[i, j]
                U_r[i, j] = orig + 0.5
                if np.linalg.norm(np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r) - T_target) < 1e-9:
                    return U_r, V_r, W_r
                U_r[i, j] = orig - 0.5
                if np.linalg.norm(np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r) - T_target) < 1e-9:
                    return U_r, V_r, W_r
                U_r[i, j] = orig
            elif mat_idx == 1:
                i, j = np.random.randint(V_r.shape[0]), np.random.randint(V_r.shape[1])
                orig = V_r[i, j]
                V_r[i, j] = orig + 0.5
                if np.linalg.norm(np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r) - T_target) < 1e-9:
                    return U_r, V_r, W_r
                V_r[i, j] = orig - 0.5
                if np.linalg.norm(np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r) - T_target) < 1e-9:
                    return U_r, V_r, W_r
                V_r[i, j] = orig
            else:
                i, j = np.random.randint(W_r.shape[0]), np.random.randint(W_r.shape[1])
                orig = W_r[i, j]
                W_r[i, j] = orig + 0.5
                if np.linalg.norm(np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r) - T_target) < 1e-9:
                    return U_r, V_r, W_r
                W_r[i, j] = orig - 0.5
                if np.linalg.norm(np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r) - T_target) < 1e-9:
                    return U_r, V_r, W_r
                W_r[i, j] = orig

    return U_r, V_r, W_r

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    T_target = _construct_tensor(n, m, p)
    
    # Theoretical lower bound is roughly 20. We try from 29 down to 20.
    # 30 is the schoolbook rank.
    ranks_to_try = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    
    for rank in ranks_to_try:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts for ALS
        # We use a mix of pure random and "near-schoolbook" initializations
        max_restarts = 100
        found = False
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            try:
                # Strategy: 
                # 1. Random init
                # 2. If attempt > 50, use a perturbed schoolbook init (projected to rank)
                
                init = None
                if attempt > 50:
                    # Create a random rank-rank decomposition of the schoolbook tensor
                    # This is complex, so we just use random init but with smaller variance
                    # to encourage convergence to sparse-like structures?
                    # Actually, let's stick to robust random init with different seeds.
                    pass
                
                np.random.seed(attempt * 1000 + rank)
                U, V, W = _als_decompose(T_target, rank, max_iter=300, tol=1e-12, init=init)
                
                # Round to half-integers
                U_r, V_r, W_r = _local_search_rounding(U, V, W, T_target, n, m, p, max_steps=20)
                
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
