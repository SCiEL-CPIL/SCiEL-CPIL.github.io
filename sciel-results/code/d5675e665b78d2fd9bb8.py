import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _khatri_rao(A, B):
    """Vectorized Khatri-Rao product: A is (a, R), B is (b, R) -> result is (a*b, R)
    where result[a_idx * b + b_idx, r] = A[a_idx, r] * B[b_idx, r]"""
    a, R = A.shape
    b, _ = B.shape
    # Reshape to (a, 1, R) and (1, b, R), multiply, then reshape to (a*b, R)
    return (A[:, np.newaxis, :] * B[np.newaxis, :, :]).reshape(a * b, R)

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the tensor T
    # T is dim_u x dim_v x dim_w
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Unfold T
    T1 = T.reshape(dim_u, dim_v * dim_w)
    T2 = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
    T3 = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook))
    V_school = np.zeros((dim_v, rank_schoolbook))
    W_school = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 120  # Full budget with safety margin
    
    def _als_solve(R, init_type, seed=0):
        """Run ALS for given rank R with specified initialization type.
        Returns (U, V, W, converged) or None on failure."""
        if init_type == 0:
            # Random Gaussian
            rng = np.random.RandomState(seed)
            U = rng.randn(dim_u, R)
            V = rng.randn(dim_v, R)
            W = rng.randn(dim_w, R)
        elif init_type == 1:
            # SVD-based initialization from T1
            try:
                U_svd, _, _ = np.linalg.svd(T1, full_matrices=False)
                U = U_svd[:, :R]
            except:
                U = np.random.randn(dim_u, R)
            rng = np.random.RandomState(seed)
            V = rng.randn(dim_v, R)
            W = rng.randn(dim_w, R)
        elif init_type == 2:
            # SVD from all three unfoldings
            try:
                U_svd, _, _ = np.linalg.svd(T1, full_matrices=False)
                U = U_svd[:, :R]
            except:
                U = np.random.randn(dim_u, R)
            try:
                V_svd, _, _ = np.linalg.svd(T2, full_matrices=False)
                V = V_svd[:, :R]
            except:
                V = np.random.randn(dim_v, R)
            try:
                W_svd, _, _ = np.linalg.svd(T3, full_matrices=False)
                W = W_svd[:, :R]
            except:
                W = np.random.randn(dim_w, R)
        else:
            # Structured random with sign patterns
            rng = np.random.RandomState(seed)
            U = rng.choice([-1, 1], size=(dim_u, R)) * rng.rand(dim_u, R)
            V = rng.choice([-1, 1], size=(dim_v, R)) * rng.rand(dim_v, R)
            W = rng.choice([-1, 1], size=(dim_w, R)) * rng.rand(dim_w, R)
            
        max_iter = 200
        tol = 1e-10
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
                
            # Update U: T1 = U @ M.T, M = _khatri_rao(V, W)
            M = _khatri_rao(V, W)
            try:
                U_new = T1 @ np.linalg.pinv(M)
            except:
                return None
                
            # Update V: T2 = V @ N.T, N = _khatri_rao(U, W)
            N = _khatri_rao(U_new, W)
            try:
                V_new = T2 @ np.linalg.pinv(N)
            except:
                return None
                
            # Update W: T3 = W @ P.T, P = _khatri_rao(U, V)
            P = _khatri_rao(U_new, V_new)
            try:
                W_new = T3 @ np.linalg.pinv(P)
            except:
                return None
                
            # Check convergence
            diff_u = np.linalg.norm(U_new - U, 'fro') / (np.linalg.norm(U, 'fro') + 1e-15)
            diff_v = np.linalg.norm(V_new - V, 'fro') / (np.linalg.norm(V, 'fro') + 1e-15)
            diff_w = np.linalg.norm(W_new - W, 'fro') / (np.linalg.norm(W, 'fro') + 1e-15)
            
            U, V, W = U_new, V_new, W_new
            
            if max(diff_u, diff_v, diff_w) < tol:
                # Verify
                T_recon = np.zeros((dim_u, dim_v, dim_w))
                for r in range(R):
                    T_recon += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
                if np.allclose(T_recon, T, atol=1e-7):
                    return (U, V, W, True)
                else:
                    return (U, V, W, False)
        
        # Final verification even if not converged
        T_recon = np.zeros((dim_u, dim_v, dim_w))
        for r in range(R):
            T_recon += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        if np.allclose(T_recon, T, atol=1e-7):
            return (U, V, W, True)
        return (U, V, W, False)
    
    # Try ranks from 35 down to 23
    for R in range(rank_schoolbook - 1, 22, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple initialization strategies and restarts
        found = False
        for init_type in range(4):
            for restart in range(8):
                if time.time() > deadline - 10:
                    break
                    
                result = _als_solve(R, init_type, seed=restart * 1000 + init_type)
                if result is not None:
                    U, V, W, converged = result
                    if converged:
                        best_rank = R
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        found = True
                        break
            if found:
                break
        
        if found:
            # Continue searching for lower rank
            continue
    
    return best_result
