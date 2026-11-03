import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin for 150s limit

def _build_target_tensor(n, m, p):
    """Construct the target matrix multiplication tensor."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p, T_target, tol=1e-6):
    """
    Verify if U, V, W form a valid decomposition of T_target.
    Checks if entries are integer or half-integer and if the decomposition is exact.
    """
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    # Check if entries are close to integers or half-integers
    # We check if 2*U, 2*V, 2*W are close to integers.
    U_2 = U * 2
    V_2 = V * 2
    W_2 = W * 2
    
    U_round = np.round(U_2)
    V_round = np.round(V_2)
    W_round = np.round(W_2)
    
    # Check distance to nearest half-integer
    if not (np.allclose(U_2, U_round, atol=1e-4) and
            np.allclose(V_2, V_round, atol=1e-4) and
            np.allclose(W_2, W_round, atol=1e-4)):
        return False

    # Snap to half-integers
    U_snap = U_round / 2.0
    V_snap = V_round / 2.0
    W_snap = W_round / 2.0
    
    # Exact check
    T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
        
    return True, U_snap, V_snap, W_snap

def _als_decompose(T_target, rank, max_iters=50, tol=1e-8, seed=0):
    """
    Perform Alternating Least Squares to find a rank-`rank` decomposition.
    Returns U, V, W or None if it fails to converge or if time is up.
    """
    dim_U, dim_V, dim_W = T_target.shape
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(dim_U, rank) * 0.1
    V = rng.randn(dim_V, rank) * 0.1
    W = rng.randn(dim_W, rank) * 0.1
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            return None
            
        # Solve for U: Fix V, W.
        # T_a, b, c = sum_r U_a, r * V_b, r * W_c, r
        # For fixed a, sum_r U_a, r * (V_b, r * W_c, r) = T_a, b, c
        # Let M_U be a matrix where row a is the vector of (V_b, r * W_c, r) flattened.
        # We can solve this as a linear system for each a, or batch it.
        
        # Construct matrix A_U such that A_U[a, r] = sum_{b,c} ... no, that's not right.
        # The equation is: sum_r U[a, r] * (V[b, r] * W[c, r]) = T[a, b, c]
        # For a fixed a, we have a linear system in U[a, :]:
        # sum_r U[a, r] * (V[:, r] * W[:, r])_flattened = T[a, :, :].flattened
        # Let M_VW[r, b, c] = V[b, r] * W[c, r].
        # Then sum_r U[a, r] * M_VW[r, b, c] = T[a, b, c].
        # This is equivalent to: (M_VW_reshaped @ U.T)[a, :] = T[a, :, :].flattened
        # where M_VW_reshaped is of shape (dim_V * dim_W, rank) with elements M_VW[r, b, c] at index (b*dim_W+c, r).
        
        M_VW = V[:, :, None] * W[None, :, :] # shape (dim_V, dim_W, rank)
        M_VW_flat = M_VW.transpose(2, 0, 1).reshape(rank, -1) # shape (rank, dim_V * dim_W)
        # We want U such that U @ M_VW_flat.T = T.reshape(dim_U, -1)
        # U @ A = B => U = B @ A^T (A A^T)^-1 ? No, U is (dim_U, rank).
        # Let X = T.reshape(dim_U, -1). Shape (dim_U, dim_V*dim_W).
        # Equation: sum_r U[a, r] * A[r, j] = X[a, j] for all a, j.
        # This is X = U @ A.T.
        # So U = X @ (A.T A)^-1 A.T ? No, that's for U as a vector.
        # For each row a of U, we solve: U[a, :] @ A.T = X[a, :]
        # U[a, :] = X[a, :] @ (A.T A)^-1 A.T is not correct if A.T is not square.
        # Actually, U[a, :] is the solution to min || U[a, :] @ A.T - X[a, :] ||_F^2
        # This is a least squares problem: A @ U[a, :].T = X[a, :].T
        # So U[a, :].T = np.linalg.lstsq(A, X[a, :].T, rcond=None)[0]
        
        A = M_VW_flat.T # Shape (dim_V * dim_W, rank)
        X = T_target.reshape(dim_U, -1) # Shape (dim_U, dim_V * dim_W)
        
        # Solve U.T = A^+ X.T
        # U.T is (rank, dim_U). X.T is (dim_V*dim_W, dim_U).
        # We solve A @ U.T = X.T
        try:
            U_new = np.linalg.lstsq(A, X.T, rcond=None)[0].T
        except Exception:
            return None
        U = U_new
        
        # Solve for V: Fix U, W.
        # sum_r V[b, r] * (U[a, r] * W[c, r]) = T[a, b, c]
        M_UW = U[:, :, None] * W[None, :, :] # shape (dim_U, dim_W, rank)
        M_UW_flat = M_UW.transpose(2, 0, 1).reshape(rank, -1) # shape (rank, dim_U * dim_W)
        A = M_UW_flat.T # Shape (dim_U * dim_W, rank)
        X = T_target.transpose(1, 0, 2).reshape(dim_V, -1) # Shape (dim_V, dim_U * dim_W)
        
        try:
            V_new = np.linalg.lstsq(A, X.T, rcond=None)[0].T
        except Exception:
            return None
        V = V_new
        
        # Solve for W: Fix U, V.
        # sum_r W[c, r] * (U[a, r] * V[b, r]) = T[a, b, c]
        M_UV = U[:, :, None] * V[None, :, :] # shape (dim_U, dim_V, rank)
        M_UV_flat = M_UV.transpose(2, 0, 1).reshape(rank, -1) # shape (rank, dim_U * dim_V)
        A = M_UV_flat.T # Shape (dim_U * dim_V, rank)
        X = T_target.transpose(2, 0, 1).reshape(dim_W, -1) # Shape (dim_W, dim_U * dim_V)
        
        try:
            W_new = np.linalg.lstsq(A, X.T, rcond=None)[0].T
        except Exception:
            return None
        W = W_new
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_recon - T_target)
        if residual < tol:
            break
            
    return U, V, W

def _try_rank(n, m, p, T_target, rank, max_seeds=50, seed_base=1000):
    """
    Attempt to find a decomposition of the given rank using ALS.
    """
    for seed in range(max_seeds):
        if time.time() > _DEADLINE:
            return None
            
        result = _als_decompose(T_target, rank, max_iters=100, seed=seed_base + seed)
        if result is None:
            continue
            
        U, V, W = result
        
        # Verify and snap
        ver_res = _verify_exact(U, V, W, n, m, p, T_target)
        if ver_res:
            return ver_res[1], ver_res[2], ver_res[3]
            
    return None

def _get_schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    n, m, p = 2, 3, 4
    
    T_target = _build_target_tensor(n, m, p)
    
    # Try rank 23
    # This is the known optimal rank for 2x3x4 matrix multiplication
    res_23 = _try_rank(n, m, p, T_target, 23, max_seeds=100, seed_base=100)
    if res_23 is not None:
        U, V, W = res_23
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook (Rank 24)
    U, V, W = _get_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
