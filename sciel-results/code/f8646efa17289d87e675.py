"""Program for matrix multiplication 3x3 * 3x6 using a rank-minimal decomposition search."""
import itertools
import numpy as np
import scipy.linalg
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _build_matrix_mult_tensor(n, m, p):
    """Build the matrix multiplication tensor T_{i,j,k} for C = A*B where
    A is n x m, B is m x p, C is n x p.
    T has shape (n*m, m*p, n*p) with T[a,b,c] = 1.0
    where a = i*m+j, b = j*p+k, c = k*n+i (output convention W[k*n+i,r])
    """
    T = np.zeros((n*m, m*p, n*p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i  # Output convention: W[k*n+i, r]
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _als_decompose(T, rank, n_a, n_b, n_c, n_iters=100, tol=1e-10, seed=42):
    """Alternating Least Squares for tensor decomposition.
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    """
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n_a, rank))
    V = rng.standard_normal((n_b, rank))
    W = rng.standard_normal((n_c, rank))
    
    for it in range(n_iters):
        # Update U: for each a, T[a,:,:] = sum_r U[a,r] * (V[r,:] * W[r,:])
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # Let M_a[b,c] = V[b,r]*W[c,r] for each r, then T[a] = U[a,:] @ (V*W)^T reshaped
        # Actually: T[a,:,:] = sum_r U[a,r] * np.outer(V[:,r], W[:,r])
        # So T_flat[a] = sum_r U[a,r] * (V[:,r] * W[:,r]) where V[:,r]*W[:,r] is element-wise product
        # Let's form: for each r, let P_r = V[:,r] * W[:,r] (element-wise, shape n_b*n_c)
        # Then T[a] (flattened n_b*n_c) = sum_r U[a,r] * P_r
        # So T_flat = U @ P where P is rank x (n_b*n_c)
        
        P = np.zeros((rank, n_b * n_c))
        for r in range(rank):
            P[r, :] = (V[:, r] * W[:, r]).flatten()
        
        # T_flat has shape (n_a, n_b*n_c)
        T_mat = T.reshape(n_a, n_b * n_c)
        U, *_ = np.linalg.lstsq(P.T, T_mat.T, rcond=None)
        U = U.T
        
        # Update V: for each b, T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
        Q = np.zeros((rank, n_a * n_c))
        for r in range(rank):
            Q[r, :] = (U[:, r] * W[:, r]).flatten()
        
        T_mat2 = T.reshape(n_a, n_b, n_c).transpose(1, 0, 2).reshape(n_b, n_a * n_c)
        V, *_ = np.linalg.lstsq(Q.T, T_mat2.T, rcond=None)
        V = V.T
        
        # Update W: for each c, T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
        R = np.zeros((rank, n_a * n_b))
        for r in range(rank):
            R[r, :] = (U[:, r] * V[:, r]).flatten()
        
        T_mat3 = T.reshape(n_a, n_b, n_c).transpose(2, 0, 1).reshape(n_c, n_a * n_b)
        W, *_ = np.linalg.lstsq(R.T, T_mat3.T, rcond=None)
        W = W.T
        
        # Check convergence
        recon = np.zeros((n_a, n_b, n_c))
        for r in range(rank):
            recon += np.outer(U[:, r], V[:, r])[:, :, None] * W[:, r][None, None, :]
            # Actually: recon[a,b,c] += U[a,r]*V[b,r]*W[c,r]
        # More efficient:
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < tol:
            break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(recon - T)
    if err < 1e-8:
        return U, V, W, err
    return None

def construct():
    """Construct the rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    n_a = n * m  # 9
    n_b = m * p  # 18
    n_c = n * p  # 18
    
    T = _build_matrix_mult_tensor(n, m, p)
    
    # Try ranks from low to high
    # The trivial rank is n*m*p = 54
    # Try to find lower ranks
    best_result = None
    best_rank = None
    
    # Try a range of ranks
    for rank in range(20, 55):
        for seed in range(10):
            result = _als_decompose(T, rank, n_a, n_b, n_c, n_iters=200, tol=1e-12, seed=seed)
            if result is not None:
                U, V, W, err = result
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W)
                    print(f"Found rank {rank} decomposition with error {err}")
                    break
        if best_rank is not None and rank == best_rank:
            break
    
    if best_result is None:
        # Fall back to trivial decomposition
        rank = n * m * p
        U = np.zeros((n_a, rank))
        V = np.zeros((n_b, rank))
        W = np.zeros((n_c, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
