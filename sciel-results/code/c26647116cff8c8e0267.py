import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0

    nm, mp, pn = n * m, m * p, p * n
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition(U, V, W, tol=1e-9):
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        T_full = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
        return np.allclose(T_full, T_target, atol=tol)

    def project_to_half_integer(U, V, W):
        U_h = np.round(2.0 * np.asarray(U)) / 2.0
        V_h = np.round(2.0 * np.asarray(V)) / 2.0
        W_h = np.round(2.0 * np.asarray(W)) / 2.0
        return U_h, V_h, W_h

    def optimize_rank(U, V, W, max_iter=500, tol=1e-12):
        """
        Alternating least squares to refine a decomposition.
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Update U: T = U (V^T \otimes W^T)
            # Reshape T to (nm, mp*pn)
            T_flat = T_target.reshape(nm, mp * pn)
            VW = np.einsum('vr,wr->r,vw', V, W).reshape(mp * pn, -1)
            # T_flat = U @ VW
            # U = T_flat @ VW^T (VW^T VW)^-1
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
                
            # Update V
            WU = np.einsum('wr,ur->r,wu', W, U).reshape(pn * nm, -1)
            T_flat_v = T_target.reshape(mp, pn * nm)
            try:
                V_new, _, _, _ = np.linalg.lstsq(WU, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
                
            # Update W
            UV = np.einsum('ur,vr->r,uv', U, V).reshape(nm * mp, -1)
            T_flat_w = T_target.reshape(pn, nm * mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence
            T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T_full - T_target)
            if err < tol:
                break
                
        return U, V, W

    def try_find_lower_rank(target_rank, n_restarts=10):
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 10:
                return None
                
            np.random.seed(123 + restart)
            
            # Initialize with schoolbook and random perturbations
            U0, V0, W0 = np.zeros((nm, target_rank)), np.zeros((mp, target_rank)), np.zeros((pn, target_rank))
            
            # Fill with random projections of schoolbook
            # Or just random init
            U0 = np.random.randn(nm, target_rank) * 0.1
            V0 = np.random.randn(mp, target_rank) * 0.1
            W0 = np.random.randn(pn, target_rank) * 0.1
            
            # Optimize
            U, V, W = optimize_rank(U0, V0, W0, max_iter=100)
            
            if check_decomposition(U, V, W, tol=1e-6):
                U_h, V_h, W_h = project_to_half_integer(U, V, W)
                if check_decomposition(U_h, V_h, W_h, tol=1e-9):
                    return {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
                    
        return None

    # Try rank 41
    result = try_find_lower_rank(41, n_restarts=20)
    if result:
        return result

    # Try rank 40
    if time.time() - start_time < time_limit - 20:
        result = try_find_lower_rank(40, n_restarts=10)
        if result:
            return result

    return get_schoolbook()
