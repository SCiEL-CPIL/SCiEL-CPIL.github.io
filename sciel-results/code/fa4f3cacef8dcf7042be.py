import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Build the target tensor T
    # T[u, v, w] = 1 if there exist i, j, k such that:
    # u = i*m + j, v = j*p + k, w = k*n + i
    # Otherwise 0
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def verify(U, V, W, tol=1e-9):
        """
        Check if U, V, W exactly decompose T.
        """
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for r in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Check if T_calc matches T
        if not np.allclose(T_calc, T, atol=tol):
            return False
        
        # Check if entries are integer or half-integer
        def is_half_int(arr):
            doubled = arr * 2
            if not np.allclose(doubled, np.round(doubled), atol=1e-9):
                return False
            return True
        
        return is_half_int(U) and is_half_int(V) and is_half_int(W)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_and_verify(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if verify(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def als_decompose(rank, max_iter=200, tol=1e-12, U_init=None, V_init=None, W_init=None):
        """
        Alternating Least Squares for tensor decomposition.
        """
        if U_init is None:
            U = np.random.randn(U_size, rank) * 0.5
        else:
            U = U_init.copy()
        if V_init is None:
            V = np.random.randn(V_size, rank) * 0.5
        else:
            V = V_init.copy()
        if W_init is None:
            W = np.random.randn(W_size, rank) * 0.5
        else:
            W = W_init.copy()
        
        prev_cost = np.inf
        for it in range(max_iter):
            # Update U
            # T reshaped to (U_size, V_size*W_size)
            # M_VW (V_size*W_size, rank) where M_VW[v*W_size+w, r] = V[v, r] * W[w, r]
            M_VW = np.einsum('vr,wr->vwr', V, W).reshape(V_size * W_size, rank)
            # Solve M_VW @ U.T = T_flat.T
            T_flat = T.reshape(U_size, V_size * W_size)
            try:
                U_new = np.linalg.lstsq(M_VW, T_flat.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None, prev_cost
            
            # Update V
            # T reshaped to (V_size, U_size*W_size)
            # M_UW (U_size*W_size, rank) where M_UW[u*W_size+w, r] = U[u, r] * W[w, r]
            M_UW = np.einsum('ur,wr->uwr', U_new, W).reshape(U_size * W_size, rank)
            T_flat_v = T.transpose(1, 0, 2).reshape(V_size, U_size * W_size)
            try:
                V_new = np.linalg.lstsq(M_UW, T_flat_v.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None, prev_cost
                
            # Update W
            # T reshaped to (W_size, U_size*V_size)
            # M_UV (U_size*V_size, rank) where M_UV[u*V_size+v, r] = U_new[u, r] * V_new[v, r]
            M_UV = np.einsum('ur,vr->uvr', U_new, V_new).reshape(U_size * V_size, rank)
            T_flat_w = T.transpose(2, 0, 1).reshape(W_size, U_size * V_size)
            try:
                W_new = np.linalg.lstsq(M_UV, T_flat_w.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None, prev_cost

            # Calculate cost
            T_calc = np.zeros((U_size, V_size, W_size))
            for r in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U_new[:, r], V_new[:, r], W_new[:, r])
            cost = np.sum((T_calc - T)**2)
            
            if abs(prev_cost - cost) < tol:
                break
                
            U, V, W = U_new, V_new, W_new
            prev_cost = cost
            
        return (U, V, W), cost

    # Fallback to schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = schoolbook()
    np.random.seed(42)
    
    def sparse_init(size, rank, density=0.3):
        """Generate sparse random initialization with values in {-1, 0, 1}"""
        M = np.zeros((size, rank))
        for i in range(rank):
            # Randomly select indices to be non-zero
            idx = np.random.choice(size, int(size * density), replace=False)
            M[idx, i] = np.random.choice([-1, 1], size=len(idx))
        return M

    # Try to find rank 14 (unlikely but fast check)
    target_rank = 14
    for r in range(20):
        if time.time() > deadline - 5:
            break
        U_init = sparse_init(U_size, target_rank)
        V_init = sparse_init(V_size, target_rank)
        W_init = sparse_init(W_size, target_rank)
        
        res, cost = als_decompose(target_rank, U_init=U_init, V_init=V_init, W_init=W_init, max_iter=50)
        
        if res is not None:
            U_sol, V_sol, W_sol = res
            snapped = snap_and_verify(U_sol, V_sol, W_sol)
            if snapped:
                U_f, V_f, W_f = snapped
                best = {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                break

    # If rank 14 failed, try rank 15 (expected minimal rank)
    if best["U"][0].__len__() == 18:
        target_rank = 15
        # Use more restarts and sparse init for rank 15
        max_restarts = 300
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            # Vary density slightly
            density = 0.2 + 0.1 * (r % 3)
            U_init = sparse_init(U_size, target_rank, density)
            V_init = sparse_init(V_size, target_rank, density)
            W_init = sparse_init(W_size, target_rank, density)
            
            res, cost = als_decompose(target_rank, U_init=U_init, V_init=V_init, W_init=W_init, max_iter=50)
            
            if res is not None:
                U_sol, V_sol, W_sol = res
                snapped = snap_and_verify(U_sol, V_sol, W_sol)
                if snapped:
                    U_f, V_f, W_f = snapped
                    best = {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                    break

    # If rank 15 failed, try rank 16
    if best["U"][0].__len__() == 18:
        target_rank = 16
        for r in range(100):
            if time.time() > deadline - 5:
                break
            U_init = sparse_init(U_size, target_rank, 0.25)
            V_init = sparse_init(V_size, target_rank, 0.25)
            W_init = sparse_init(W_size, target_rank, 0.25)
            
            res, cost = als_decompose(target_rank, U_init=U_init, V_init=V_init, W_init=W_init, max_iter=50)
            
            if res is not None:
                U_sol, V_sol, W_sol = res
                snapped = snap_and_verify(U_sol, V_sol, W_sol)
                if snapped:
                    U_f, V_f, W_f = snapped
                    best = {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                    break

    return best
# EVOLVE-BLOCK-END
