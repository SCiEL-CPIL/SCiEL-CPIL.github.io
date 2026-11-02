"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 3, 3]
DEADLINE_SECONDS = 115  # Safety margin within 120s limit


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor M of shape (n*m, m*p, p*n).
    M[u,v,w] = 1 if u=(i,j), v=(j,k), w=(k,i) for valid indices, else 0."""
    M = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                M[u, v, w] = 1.0
    return M


def _verify_decomposition(U, V, W, M, tol=1e-9):
    """Check if sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] ≈ M."""
    rank = U.shape[1]
    reconstruction = np.zeros_like(M)
    for c in range(rank):
        reconstruction += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    diff = np.abs(reconstruction - M)
    return diff.max() <= tol


def _snap_to_half_int(arr):
    """Snap values to nearest half-integer."""
    return np.round(arr * 2) / 2.0


def _search_rank(R, M, time_budget, seed=None):
    """Try to find a rank-R decomposition via iterative fitting.
    
    Uses an approach where we initialize and then refine using
    the structure: M_{u,v,w} = sum_c U[u,c]*V[v,c]*W[w,c]
    
    We alternate: fix U,V, solve for W (linear), then fix V,W solve for U, etc.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n, m, p = M.shape[0] // 3, 3, 3  # For 2x3x3: n*m=6, m*p=9, p*n=6
    # Actually M.shape = (n*m, m*p, p*n) = (6, 9, 6)
    nm, mp, pn = M.shape
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    num_restarts = min(200, max(10, int(time_budget / 0.5)))
    
    for restart in range(num_restarts):
        t_start = time.time()
        if time.time() - t_start > time_budget:
            break
        
        # Initialize with random values
        U = np.random.randn(nm, R) * 0.5
        V = np.random.randn(mp, R) * 0.5
        W = np.random.randn(pn, R) * 0.5
        
        # Iterate: alternate solving for each factor
        for it in range(200):
            # Solve for W: for each (u,v), M[u,v,:] = W^T @ (U[:,c]*V[v,c] for c)
            # M[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # For fixed U,V: M[u,v,:] = (U[:,c]*V[v,c])_c dot W[:,c] for each w
            # i.e., M[u,v] = W @ diag(U[:,c]*V[v,c]) ... not quite
            # M[u,v,w] = sum_c (U[u,c]*V[v,c]) * W[w,c]
            # So M[u,v,:] = W.T @ (U[u,:]*V[v,:])  ... no
            # M[u,v,:] where the : is over w: M[u,v,w] = sum_c U[u,c]V[v,c]W[w,c]
            # Let A_uv_c = U[u,c]*V[v,c], then M[u,v,w] = sum_c A_uv_c * W[w,c]
            # So M[u,v,:] = (A_uv @ W.T).T ... 
            # M[u,v] (length pn) = W.T @ A_uv  where A_uv is length R
            # Actually: M[u,v,w] = sum_c A_uv[c] * W[w,c] = (W.T @ A_uv)[w]... no
            # sum_c A_uv[c] * W[w,c] = A_uv @ W[:, w] for each w
            # So the vector M[u,v,:] = A_uv @ W (as a row) = (W.T @ A_uv).T
            # To solve for W given U,V:
            # min_W sum_{u,v} ||M[u,v,:] - A_uv @ W||^2
            # This is a linear least squares problem.
            
            # Flatten: for each (u,v), we have pn equations.
            # A_uv is R-vector, W is R x pn matrix.
            # M[u,v] = A_uv @ W (row vector of length pn)
            # Stack all (u,v): let A be (nm*mp, R), B be (nm*mp, pn)
            # B = A @ W  (where each row of A is A_uv for some (u,v))
            # W = (A^T A)^{-1} A^T B (least squares)
            
            A = (U[:, :, None] * V[None, :, :]).reshape(-1, R)  # (nm*mp, R)
            B = M.reshape(-1, pn)  # (nm*mp, pn)
            
            try:
                W, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            except np.linalg.LinAlgError:
                break
            
            # Solve for U similarly
            # M[u,v,w] = sum_c U[u,c] * (V[v,c]*W[w,c])
            # For fixed V,W: M[u,v,w] = U[u,:] @ (V[v,:]*W[w,:])
            # M[u] (shape mp*pn) = U[u,:] @ (V*W)^T ... 
            # Let C_vw_c = V[v,c]*W[w,c], then M[u,v,w] = sum_c U[u,c]*C_vw[c]
            # M[u, v, w] = U[u,:] dot C_vw
            # Flatten over (v,w): M[u,:] = U[u,:] @ C^T where C is (mp*pn, R)
            C = (V[:, :, None] * W[None, :, :]).reshape(-1, R)  # (mp*pn, R)
            B_u = M.reshape(nm, -1)  # (nm, mp*pn)
            try:
                U, _, _, _ = np.linalg.lstsq(C.T, B_u, rcond=None)
            except np.linalg.LinAlgError:
                break
            
            # Solve for V
            # M[u,v,w] = sum_c (U[u,c]*W[w,c]) * V[v,c]
            # For fixed U,W: D_uk_c = U[u,c]*W[w,c]
            # M[u,v,w] = D_uk @ V[v,:] ... 
            # M[u, v, w] = sum_c D_uk[c] * V[v,c]
            # Flatten over (u,w): M[:, v, :] = V[v,:] @ D^T where D is (nm*pn, R)
            D = (U[:, :, None] * W[None, :, :]).reshape(-1, R)  # (nm*pn, R)
            B_v = M.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
            try:
                V, _, _, _ = np.linalg.lstsq(D.T, B_v, rcond=None)
            except np.linalg.LinAlgError:
                break
            
            # Compute error
            recon = np.zeros_like(M)
            for c in range(R):
                recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            err = np.abs(recon - M).max()
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-7:
                break
        
        # Check if we found a good solution
        if best_err < 1e-6:
            break
    
    if best_err < 1e-5:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err


def _try_half_integer_rank(R, M, time_budget):
    """Try to find a rank-R decomposition with half-integer entries."""
    nm, mp, pn = M.shape
    best_err = float('inf')
    best = None
    
    # Random search over half-integer entries in a bounded range
    max_val = 4  # Try values in {-4, -3.5, ..., 3.5, 4}
    values = [i * 0.5 for i in range(-8, 9)]
    
    t_start = time.time()
    num_attempts = 0
    max_attempts = 50000
    
    while num_attempts < max_attempts and (time.time() - t_start) < time_budget:
        num_attempts += 1
        
        # Random initialization with half-integers
        U = np.array([[random.choice(values) for _ in range(R)] for _ in range(nm)])
        V = np.array([[random.choice(values) for _ in range(R)] for _ in range(mp)])
        W = np.array([[random.choice(values) for _ in range(R)] for _ in range(pn)])
        
        recon = np.zeros_like(M)
        for c in range(R):
            recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        err = np.abs(recon - M).max()
        
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-9:
            break
    
    if best_err < 1e-9:
        return best[0], best[1], best[2], best_err
    return None, None, None, best_err


def _known_rank17():
    """Attempt to construct a known rank-17 decomposition.
    
    For 2x3x3 matrix multiplication, the tensor rank is known to be 17.
    Here we try to construct it explicitly using the structure of the problem.
    """
    n, m, p = 2, 3, 3
    nm, mp, pn = n * m, m * p, p * n  # 6, 9, 6
    
    # The schoolbook has rank 18. To get rank 17, we need to merge two rank-1 terms.
    # This is non-trivial. Let's use a known construction or fall back.
    # 
    # One approach: use the fact that for small matrices, rank reductions
    # come from exploiting linear dependencies.
    #
    # For now, return None to indicate we couldn't find one analytically.
    return None


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    M = _build_tensor(n, m, p)
    
    t_global_start = time.time()
    remaining = lambda: DEADLINE_SECONDS - (time.time() - t_global_start)
    
    # First, try to find rank 17
    # Strategy 1: Numerical search
    if remaining() > 5:
        U, V, W, err = _search_rank(17, M, time_budget=min(remaining() - 3, 60), seed=42)
        if U is not None and err < 1e-6:
            # Snap to nearest half-integer and verify
            U_s = _snap_to_half_int(U)
            V_s = _snap_to_half_int(V)
            W_s = _snap_to_half_int(W)
            if _verify_decomposition(U_s, V_s, W_s, M, tol=1e-9):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            # If snapping didn't work, try without snapping but check if values are close to half-ints
            if _verify_decomposition(U, V, W, M, tol=1e-9):
                # Values are valid but may not be half-integers; round them
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                if _verify_decomposition(U_r, V_r, W_r, M, tol=1e-9):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Strategy 2: Random half-integer search for rank 17
    if remaining() > 5:
        U, V, W, err = _try_half_integer_rank(17, M, time_budget=min(remaining() - 3, 30))
        if U is not None and err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 3: Try rank 18 (schoolbook) as fallback - guaranteed to work
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
