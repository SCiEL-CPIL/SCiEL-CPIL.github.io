import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T as a flat vector of non-zero entries and indices
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nonzero_indices.append((u_idx, v_idx, w_idx))
    
    nonzero_set = set(nonzero_indices)
    
    # Precompute the target tensor as a 3D array
    T = np.zeros((dim_u, dim_v, dim_w))
    for u_idx, v_idx, w_idx in nonzero_indices:
        T[u_idx, v_idx, w_idx] = 1.0
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        # Check non-zero entries
        for u_idx, v_idx, w_idx in nonzero_indices:
            val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
            if abs(val - 1.0) > tol:
                return False
        # Check all zero entries
        for u_idx in range(dim_u):
            for v_idx in range(dim_v):
                for w_idx in range(dim_w):
                    if (u_idx, v_idx, w_idx) not in nonzero_set:
                        val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                        if abs(val) > tol:
                            return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap entries to nearest half-integer."""
        def snap_matrix(M):
            M_snapped = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    best_k = 0
                    best_err = abs(val - 0)
                    for k in range(-40, 41):
                        candidate = k / 2.0
                        err = abs(val - candidate)
                        if err < best_err:
                            best_err = err
                            best_k = k
                    M_snapped[i, j] = best_k / 2.0
            return M_snapped
        return snap_matrix(U), snap_matrix(V), snap_matrix(W)

    def snap_to_integers(U, V, W):
        """Snap entries to nearest integer."""
        def snap_matrix(M):
            return np.round(M)
        return snap_matrix(U), snap_matrix(V), snap_matrix(W)

    def run_als_vectorized(U, V, W, max_iters=200, tol=1e-12):
        """Fast vectorized ALS."""
        r = U.shape[1]
        prev_err = float('inf')
        
        for it in range(max_iters):
            # Update U: for each u_idx, solve A @ U[u_idx,:] = b
            # A has shape (dim_v * dim_w, r), A[v_idx * dim_w + w_idx, c] = V[v_idx, c] * W[w_idx, c]
            VW = np.einsum('vc,wc->vwc', V, W).reshape(dim_v * dim_w, r)
            for u_idx in range(dim_u):
                b = T[u_idx, :, :].flatten()
                U[u_idx, :] = np.linalg.lstsq(VW, b, rcond=None)[0]
            
            # Update V: for each v_idx, solve A @ V[v_idx,:] = b
            # A has shape (dim_u * dim_w, r), A[u_idx * dim_w + w_idx, c] = U[u_idx, c] * W[w_idx, c]
            UW = np.einsum('uc,wc->uwc', U, W).reshape(dim_u * dim_w, r)
            for v_idx in range(dim_v):
                b = T[:, v_idx, :].reshape(dim_u * dim_w)
                V[v_idx, :] = np.linalg.lstsq(UW, b, rcond=None)[0]
            
            # Update W: for each w_idx, solve A @ W[w_idx,:] = b
            # A has shape (dim_u * dim_v, r), A[u_idx * dim_v + v_idx, c] = U[u_idx, c] * V[v_idx, c]
            UV = np.einsum('uc,vc->uvc', U, V).reshape(dim_u * dim_v, r)
            for w_idx in range(dim_w):
                b = T[:, :, w_idx].reshape(dim_u * dim_v)
                W[w_idx, :] = np.linalg.lstsq(UV, b, rcond=None)[0]
            
            # Calculate error (sample-based for speed)
            err = 0.0
            for u_idx, v_idx, w_idx in nonzero_indices:
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                err += (val - 1.0)**2
            # Sample zero entries
            np.random.seed(it)
            for _ in range(50):
                u_idx = np.random.randint(dim_u)
                v_idx = np.random.randint(dim_v)
                w_idx = np.random.randint(dim_w)
                if (u_idx, v_idx, w_idx) not in nonzero_set:
                    val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    err += val**2
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W

    def polish(U, V, W, max_iters=50, tol=1e-14):
        """High-precision polishing pass."""
        r = U.shape[1]
        for it in range(max_iters):
            VW = np.einsum('vc,wc->vwc', V, W).reshape(dim_v * dim_w, r)
            for u_idx in range(dim_u):
                b = T[u_idx, :, :].flatten()
                U[u_idx, :] = np.linalg.lstsq(VW, b, rcond=None)[0]
            
            UW = np.einsum('uc,wc->uwc', U, W).reshape(dim_u * dim_w, r)
            for v_idx in range(dim_v):
                b = T[:, v_idx, :].reshape(dim_u * dim_w)
                V[v_idx, :] = np.linalg.lstsq(UW, b, rcond=None)[0]
            
            UV = np.einsum('uc,vc->uvc', U, V).reshape(dim_u * dim_v, r)
            for w_idx in range(dim_w):
                b = T[:, :, w_idx].reshape(dim_u * dim_v)
                W[w_idx, :] = np.linalg.lstsq(UV, b, rcond=None)[0]
        return U, V, W

    def compute_max_err(U, V, W):
        """Compute max absolute error over all tensor entries."""
        r = U.shape[1]
        max_err = 0.0
        for u_idx, v_idx, w_idx in nonzero_indices:
            val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
            max_err = max(max_err, abs(val - 1.0))
        for u_idx in range(dim_u):
            for v_idx in range(dim_v):
                for w_idx in range(dim_w):
                    if (u_idx, v_idx, w_idx) not in nonzero_set:
                        val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                        max_err = max(max_err, abs(val))
        return max_err

    def structured_init(r, seed):
        """Create structured initializations that exploit tensor symmetries."""
        np.random.seed(seed)
        init_type = seed % 5
        
        if init_type == 0:
            # Random
            U = np.random.randn(dim_u, r) * 0.5
            V = np.random.randn(dim_v, r) * 0.5
            W = np.random.randn(dim_w, r) * 0.5
        elif init_type == 1:
            # Initialize based on schoolbook pattern with perturbation
            U = np.zeros((dim_u, r))
            V = np.zeros((dim_v, r))
            W = np.zeros((dim_w, r))
            for c in range(min(r, n*m*p)):
                i = c // (m*p)
                j = (c // p) % m
                k = c % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            # Perturb
            U += np.random.randn(dim_u, r) * 0.1
            V += np.random.randn(dim_v, r) * 0.1
            W += np.random.randn(dim_w, r) * 0.1
        elif init_type == 2:
            # Block structure: group by i
            U = np.zeros((dim_u, r))
            V = np.zeros((dim_v, r))
            W = np.zeros((dim_w, r))
            cols_per_i = r // n
            for i in range(n):
                for j in range(m):
                    U[i*m+j, i*cols_per_i:(i+1)*cols_per_i] = 1.0
                for k in range(p):
                    W[k*n+i, i*cols_per_i:(i+1)*cols_per_i] = 1.0
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, i*cols_per_i:(i+1)*cols_per_i] = 1.0 / np.sqrt(m*p)
            U += np.random.randn(dim_u, r) * 0.1
            V += np.random.randn(dim_v, r) * 0.1
            W += np.random.randn(dim_w, r) * 0.1
        elif init_type == 3:
            # Sparse random with structured sparsity
            U = np.zeros((dim_u, r))
            V = np.zeros((dim_v, r))
            W = np.zeros((dim_w, r))
            for c in range(r):
                # Each column activates a subset
                active_u = np.random.choice(dim_u, size=2, replace=False)
                active_v = np.random.choice(dim_v, size=3, replace=False)
                active_w = np.random.choice(dim_w, size=2, replace=False)
                for idx in active_u:
                    U[idx, c] = np.random.randn()
                for idx in active_v:
                    V[idx, c] = np.random.randn()
                for idx in active_w:
                    W[idx, c] = np.random.randn()
        else:
            # Diagonal-like initialization
            U = np.zeros((dim_u, r))
            V = np.zeros((dim_v, r))
            W = np.zeros((dim_w, r))
            for c in range(min(r, dim_u)):
                U[c, c] = 1.0
            for c in range(min(r, dim_v)):
                V[c, c] = 1.0
            for c in range(min(r, dim_w)):
                W[c, c] = 1.0
            U += np.random.randn(dim_u, r) * 0.2
            V += np.random.randn(dim_v, r) * 0.2
            W += np.random.randn(dim_w, r) * 0.2
        
        return U, V, W

    start_time = time.time()
    deadline = 130.0  # 20s safety margin before 150s kill
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Try ranks from low to high
    for r in [16, 17, 18, 19, 20, 21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
        
        # Number of restarts based on remaining time
        remaining = deadline - time.time()
        # Estimate: each ALS iteration takes ~5ms for r<=20, 200 iters = ~1s
        # So we can do roughly remaining/2 restarts
        max_restarts = min(100, max(10, int(remaining / 3)))
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            U, V, W = structured_init(r, 1000 + restart * 7)
            U, V, W = run_als_vectorized(U, V, W, max_iters=150, tol=1e-12)
            
            # Polish
            U, V, W = polish(U, V, W, max_iters=30, tol=1e-14)
            
            # Check quality
            max_err = compute_max_err(U, V, W)
            
            if max_err < 1e-6:
                # Try snapping to half-integers
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                if check_solution(U_s, V_s, W_s, tol=1e-8):
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                # Try snapping to integers
                U_si, V_si, W_si = snap_to_integers(U, V, W)
                if check_solution(U_si, V_si, W_si, tol=1e-8):
                    return {"U": U_si.tolist(), "V": V_si.tolist(), "W": W_si.tolist()}
                
                # Try mixed: snap each matrix independently to best grid
                for grid in [0.5, 1.0, 0.25]:
                    def snap_grid(M, g):
                        M_snapped = np.zeros_like(M)
                        for i in range(M.shape[0]):
                            for j in range(M.shape[1]):
                                val = M[i, j]
                                best_k = 0
                                best_err = abs(val)
                                for k in range(-20, 21):
                                    candidate = k * g
                                    err = abs(val - candidate)
                                    if err < best_err:
                                        best_err = err
                                        best_k = k
                                M_snapped[i, j] = best_k * g
                        return M_snapped
                    U_g, V_g, W_g = snap_grid(U, grid), snap_grid(V, grid), snap_grid(W, grid)
                    if check_solution(U_g, V_g, W_g, tol=1e-8):
                        return {"U": U_g.tolist(), "V": V_g.tolist(), "W": W_g.tolist()}
            
            # Track best unsnapped solution
            if best_result is None or r < best_rank:
                best_result = (U.copy(), V.copy(), W.copy(), r)
                best_rank = r
    
    # If we found a good unsnapped solution, try harder to snap
    if best_result is not None:
        U, V, W, r = best_result
        # Try multiple snapping strategies
        for _ in range(5):
            U_s, V_s, W_s = snap_to_half_integers(U, V, W)
            if check_solution(U_s, V_s, W_s, tol=1e-8):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            U_si, V_si, W_si = snap_to_integers(U, V, W)
            if check_solution(U_si, V_si, W_si, tol=1e-8):
                return {"U": U_si.tolist(), "V": V_si.tolist(), "W": W_si.tolist()}
    
    # Fallback: Schoolbook
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
