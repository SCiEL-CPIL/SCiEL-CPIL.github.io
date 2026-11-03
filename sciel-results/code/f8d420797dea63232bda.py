"""Matrix multiplication 3x3 by 3x7: rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE = 150.0  # seconds, evaluator kills at this point
SAFETY_MARGIN = 5.0  # seconds before deadline to stop and return best


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k,l] where
    (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Check if U @ V^T @ W^T reconstructs T exactly (within tolerance).
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Reconstructed tensor: sum_r U[:,r] * V[:,r] * W[:,r]
    """
    r = U.shape[1]
    # Efficient check: for each slice, verify
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Use einsum for speed
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)


def _try_rank(T, rank, max_iters=200, seed=None, time_limit=None):
    """Try to find a rank-r decomposition using iterative tensor decomposition.
    Uses a simple iterative approach: random init, then alternating minimization.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n_dim = T.shape[0]
    m_dim = T.shape[1]
    p_dim = T.shape[2]
    
    U = rng.randn(n_dim, rank)
    V = rng.randn(m_dim, rank)
    W = rng.randn(p_dim, rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Update U: for each a, U[a,:] = argmin sum_b,c T[a,b,c] V[b,:] W[c,:] - U[a,:]
        # This is a linear system: U[a,:] * (V[b,:] * W[c,:])^T = T[a,b,c]
        # More precisely: for fixed b,c, the contribution to U[a,:] is T[a,b,c] * (V[b,:] * W[c,:])
        # So U[a,:] should satisfy: sum_{b,c} T[a,b,c] * outer(V[b,:], W[c,:]) = U[a,:] * sum_{b,c} T[a,b,c] * (V[b,:] * W[c,:]) ... 
        # Actually the objective is ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
        # Gradient w.r.t. U[a,r] = -2 * sum_{b,c} T[a,b,c] V[b,r] W[c,r] + 2 * sum_{b,c} U[a,r'] V[b,r'] W[c,r'] V[b,r] W[c,r]
        # This is complex. Let's use a simpler approach: treat it as a linear regression.
        
        # Alternative: use the fact that if we fix V and W, the optimal U can be found by
        # solving a linear system. For each a:
        # T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # Let M = V * W element-wise, shape (m_dim*p_dim, rank) reshaped... 
        # Actually V[:,r] * W[:,r] is an outer product of size (m_dim, p_dim)
        # T[a,:,:] is (m_dim, p_dim)
        # So T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # This is a linear equation: T[a].flatten() = U[a,:] @ (V*W).reshape(m_dim*p_dim, rank).T
        # Wait: sum_r U[a,r] * (V[:,r] * W[:,r]) means element (b,c) is sum_r U[a,r]*V[b,r]*W[c,r]
        # So T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # Let P_r = V[:,r] * W[:,r] (element-wise product, shape m_dim x p_dim)
        # T[a,:,:] = sum_r U[a,r] * P_r
        # Flatten: T[a].flatten() = U[a,:] @ (P_1.flatten(), P_2.flatten(), ...) as columns
        # Let's build the matrix: for each r, P_r.flatten() is a vector of length m_dim*p_dim
        # So T[a].flatten() = U[a,:] @ M_a where M_a[r, :] = P_r.flatten()
        # M_a is (rank, m_dim*p_dim), so U[a,:] @ M_a.T = T[a].flatten()
        # U[a,:] = T[a].flatten() @ pinv(M_a.T)
        
        VW = V * W  # (m_dim, p_dim, rank) -> actually V is (m_dim, rank), W is (p_dim, rank)
        # V[:,r] * W[:,r] for each r gives (m_dim, p_dim)
        # Let's compute M: shape (rank, m_dim, p_dim) where M[r] = V[:,r] * W[:,r]
        M = V[:, :, None] * W[None, :, :]  # (m_dim, p_dim, rank)
        # M[b,c,r] = V[b,r]*W[c,r]
        # For each a: T[a,b,c] = sum_r U[a,r] * M[b,c,r]
        # Reshape M to (m_dim*p_dim, rank): M_flat[r, b*p_dim+c] = M[b,c,r]
        M_flat = M.transpose(2, 0, 1).reshape(rank, m_dim * p_dim)  # (rank, m_dim*p_dim)
        # T[a].flatten() = U[a,:] @ M_flat.T  => U[a,:] = T[a].flatten() @ pinv(M_flat.T)
        
        T_flat = T.reshape(n_dim, m_dim * p_dim)  # (n_dim, m_dim*p_dim)
        # Solve: T_flat @ pinv(M_flat.T) -> (n_dim, rank)
        try:
            U_new = T_flat @ np.linalg.pinv(M_flat.T)
        except np.linalg.LinAlgError:
            break
        
        # Update V
        UW = U * W  # (n_dim, p_dim, rank) where UW[a,c,r] = U[a,r]*W[c,r]
        # T[a,b,c] = sum_r V[b,r] * UW[a,c,r]
        # For each b: T[:,:,b].T.flatten() = V[b,:] @ (UW[:,:,b]).T ... 
        # T[a,c,b] = sum_r V[b,r] * U[a,r] * W[c,r]
        # T[:, :, b] is (n_dim, p_dim), and for fixed b: T[:, :, b] = V[b,:] @ (U * W[:,:,b])... 
        # Let's think: T[a,c,b] = sum_r V[b,r] * (U[a,r] * W[c,r])
        # Let Q_r = U[:,r] * W[:,r] shape (n_dim, p_dim)
        # T[:,:,b] = sum_r V[b,r] * Q_r
        # T[:,:,b].flatten() = V[b,:] @ Q_flat.T where Q_flat[r] = Q_r.flatten()
        Q = U[:, :, None] * W[None, :, :]  # (n_dim, p_dim, rank)
        Q_flat = Q.transpose(2, 0, 1).reshape(rank, n_dim * p_dim)
        T2 = T.transpose(0, 2, 1).reshape(p_dim, n_dim * m_dim)  # T2[b, a*m_dim+c] = T[a,c,b]
        # Wait, T[a,b,c], so T[:,:,b] has shape (n_dim, p_dim), indexed by (a,c)
        # T2[b, a*p_dim+c] = T[a,b,c]
        T2 = T.transpose(2, 0, 1).reshape(p_dim, n_dim * p_dim)  # (p_dim, n_dim*p_dim)
        try:
            V_new = T2 @ np.linalg.pinv(Q_flat.T)
        except np.linalg.LinAlgError:
            break
        
        # Update W
        UV = U * V  # (n_dim, m_dim, rank) where UV[a,b,r] = U[a,r]*V[b,r]
        # T[a,b,c] = sum_r W[c,r] * UV[a,b,r]
        # For each c: T[:,:,c].flatten() = W[c,:] @ (UV[:,:,c]).T...
        # T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
        # Let R_r = U[:,r] * V[:,r] shape (n_dim, m_dim)
        # T[:,:,c] = sum_r W[c,r] * R_r
        # T[:,:,c].flatten() = W[c,:] @ R_flat.T
        R = U[:, :, None] * V[None, :, :]  # (n_dim, m_dim, rank)
        R_flat = R.transpose(2, 0, 1).reshape(rank, n_dim * m_dim)
        T3 = T.transpose(2, 0, 1).reshape(p_dim, n_dim * m_dim)  # (p_dim, n_dim*m_dim)
        try:
            W_new = T3 @ np.linalg.pinv(R_flat.T)
        except np.linalg.LinAlgError:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    return best_err, best_U, best_V, best_W


def _snap_to_rational(U, V, W, max_denom=100):
    """Try to snap entries to rationals with small denominators."""
    U_s = np.zeros_like(U)
    V_s = np.zeros_like(V)
    W_s = np.zeros_like(W)
    
    for arr, arr_s in [(U, U_s), (V, V_s), (W, W_s)]:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Try to find rational p/q with q <= max_denom close to val
                best_p, best_q = int(round(val)), 1
                best_diff = abs(val - best_p / best_q)
                for q in range(2, max_denom + 1):
                    p = int(round(val * q))
                    diff = abs(val - p / q)
                    if diff < best_diff:
                        best_diff = diff
                        best_p, best_q = p, q
                arr_s[i, j] = best_p / best_q
    
    return U_s, V_s, W_s


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But for matrix multiplication, the known bounds are higher.
    # Schoolbook rank = n*m*p = 63
    # Let's try to find the minimal rank.
    
    # Start from schoolbook and try to reduce
    # First, let's try various ranks from low to high
    
    best_rank = n * m * p  # schoolbook
    best_U, best_V, best_W = None, None, None
    
    # Create schoolbook solution
    U_school = np.zeros((n * m, n * m * p))
    V_school = np.zeros((m * p, n * m * p))
    W_school = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    if _verify_decomposition(U_school, V_school, W_school, T):
        best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank decompositions
    # Known: for 2x2x2, rank is 7 (Strassen)
    # For 3x3x3, the best known is around 23
    # For 3x3x7, let's try ranks from 21 upward
    
    # Try a range of ranks
    ranks_to_try = list(range(21, min(64, n * m * p + 1)))
    
    # Sort by likelihood of success (start with lower ranks)
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        found = False
        num_restarts = 5 if rank < 40 else 3
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            err, U, V, W = _try_rank(T, rank, max_iters=100, seed=restart * 1000 + rank, 
                                      time_limit=10.0)
            
            if err < 1e-10:
                # Verify more carefully
                if _verify_decomposition(U, V, W, T, tol=1e-12):
                    found = True
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
                else:
                    # Try snapping to rationals
                    U_s, V_s, W_s = _snap_to_rational(U, V, W, max_denom=50)
                    if _verify_decomposition(U_s, V_s, W_s, T, tol=1e-12):
                        found = True
                        best_rank = rank
                        best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
                        break
        
        if found:
            # Check if we can go lower
            if rank == 21:
                break
    
    # Convert to lists for the return format
    if best_U is None:
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = n * m * p
    
    # Convert numpy arrays to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
