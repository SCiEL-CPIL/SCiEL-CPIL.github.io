import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def verify_decomposition(U, V, W, T, tol=1e-9):
        r = U.shape[1]
        # Vectorized reconstruction
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    def try_rank_structured(r, deadline, seed=0):
        """
        Try to find a rank-r decomposition using structured search.
        """
        np.random.seed(seed)
        
        # Spectral Initialization for U
        # Unfold T along mode 1: shape (dim_u, dim_v * dim_w)
        T_unfold1 = T.reshape(dim_u, dim_v * dim_w)
        try:
            U_svd, S, Vt = np.linalg.svd(T_unfold1, full_matrices=False)
            # Take top r components
            U_init = U_svd[:, :r] * S[:r]
        except:
            U_init = np.random.randn(dim_u, r) * 0.5

        # Initialize V and W randomly, scaled
        V_init = np.random.randn(dim_v, r) * 0.5
        W_init = np.random.randn(dim_w, r) * 0.5
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_error = float('inf')
        converged = False
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U
            # A_u shape: (dim_v * dim_w, r)
            A_u = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
            T_flat = T.reshape(dim_u, dim_v * dim_w)
            try:
                X, _, _, _ = np.linalg.lstsq(A_u, T_flat.T, rcond=None)
                U = X.T
            except:
                pass
                
            # Update V
            # A_v shape: (dim_u * dim_w, r)
            A_v = np.einsum('ic,kc->ikc', U, W).reshape(dim_u * dim_w, r)
            T_v_flat = np.transpose(T, (1, 0, 2)).reshape(dim_v, dim_u * dim_w)
            try:
                Y, _, _, _ = np.linalg.lstsq(A_v, T_v_flat.T, rcond=None)
                V = Y.T
            except:
                pass
                
            # Update W
            # A_w shape: (dim_u * dim_v, r)
            A_w = np.einsum('ic,jc->ijc', U, V).reshape(dim_u * dim_v, r)
            T_w_flat = np.transpose(T, (2, 0, 1)).reshape(dim_w, dim_u * dim_v)
            try:
                Z, _, _, _ = np.linalg.lstsq(A_w, T_w_flat.T, rcond=None)
                W = Z.T
            except:
                pass
                
            # Check error
            T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            error = np.linalg.norm(T_approx - T)
            
            if error < 1e-9:
                converged = True
                break
                
            if abs(prev_error - error) < 1e-14:
                break
            prev_error = error
            
        if converged:
            # Try rounding to half-integers
            for scale in [2, 4, 1]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                if verify_decomposition(U_r, V_r, W_r, T):
                    return U_r, V_r, W_r
            
            # Try integer rounding
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if verify_decomposition(U_i, V_i, W_i, T):
                return U_i, V_i, W_i
            
            # If not rounded, but error is tiny, maybe return as is? 
            # The task requires integer/half-integer entries.
            # So we must round. If rounding fails, we don't have a valid solution for this rank.
            
        return None

    deadline = time.time() + 110  # Leave 40s margin for safety and final checks
    
    best_result = None
    
    # Try rank 29
    for seed in range(10):
        if time.time() > deadline:
            break
        res = try_rank_structured(29, deadline, seed)
        if res is not None:
            best_result = res
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 28
    if time.time() < deadline:
        for seed in range(5):
            if time.time() > deadline:
                break
            res = try_rank_structured(28, deadline, seed)
            if res is not None:
                best_result = res
                break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (rank 30)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
