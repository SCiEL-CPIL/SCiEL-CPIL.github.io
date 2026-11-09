"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if this is a valid (i,j,k) triple."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrix(T, n, m, p):
    """Reshape tensor T[i,j,k] into matrix form M[i*m+j, k*p+? ... ]
    Actually we need: M[a, b] where a = i*m+j, b = j*p+k
    But the tensor is 3-way. For ALS we work with the 3-way tensor directly.
    """
    return T

def _als_decompose(T, n, m, p, rank, max_iter=200, tol=1e-8, seed=0):
    """ALS decomposition of the 3-way tensor T[i,j,k] into rank-r factors.
    U[i, c] * V[j, c] * W[k, c] summed over c.
    Returns (U, V, W) as numpy arrays or None if failed.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n, rank)
    V = rng.randn(m, rank)
    W = rng.randn(p, rank)
    
    best_err = np.inf
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Update U: for each i, solve (V^T ⊗ W^T) u_i = T[i,:,:].ravel()
        # T[i,:,:] reshaped to (m*p,), V^T ⊗ W^T is (m*p, rank)
        for i in range(n):
            # Build system: sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k]
            # For fixed i: sum_c U[i,c] * (V[:,c] * W[:,c]) = T[i,:,:]
            # Vectorize: (V ⊙ W) is (m*p, rank), each column c is V[:,c]*W[:,c]
            VW = (V * W).reshape(m * p, rank)
            rhs = T[i, :, :].ravel()
            # Solve VW^T @ u_i = rhs
            try:
                u_i, residuals, rank_r, sv = np.linalg.lstsq(VW.T, rhs, rcond=tol)
            except:
                u_i = np.zeros(rank)
            U[i, :] = u_i
        
        # Update V: for each j, solve (U^T ⊗ W^T) v_j = T[:,j,:].ravel()
        for j in range(m):
            UW = (U * W).reshape(n * p, rank)
            rhs = T[:, j, :].ravel()
            try:
                v_j, residuals, rank_r, sv = np.linalg.lstsq(UW.T, rhs, rcond=tol)
            except:
                v_j = np.zeros(rank)
            V[j, :] = v_j
        
        # Update W: for each k, solve (U^T ⊗ V^T) w_k = T[:,:,k].ravel()
        for k in range(p):
            UV = (U * V).reshape(n * m, rank)
            rhs = T[:, :, k].ravel()
            try:
                w_k, residuals, rank_r, sv = np.linalg.lstsq(UV.T, rhs, rcond=tol)
            except:
                w_k = np.zeros(rank)
            W[k, :] = w_k
        
        # Compute error
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(reconstructed - T) / np.linalg.norm(T)
        
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if err < tol:
            break
    
    if best_err < 1e-6:
        return best_factors
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We need to check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # Reconstruct T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p))
    return np.allclose(T_recon, T_target, atol=1e-9)

def _als_decompose_full(n, m, p, rank, max_iter=100, tol=1e-8, seed=0):
    """ALS on the full 3-way tensor, returning U(n*m, rank), V(m*p, rank), W(p*n, rank)."""
    T = _build_tensor(n, m, p)
    result = _als_decompose(T, n, m, p, rank, max_iter=max_iter, tol=tol, seed=seed)
    if result is None:
        return None
    
    U3, V3, W3 = result  # shapes (n,rank), (m,rank), (p,rank)
    
    # Convert to the required format:
    # U_full[i*m+j, c] = U3[i, c] for all j? No!
    # The decomposition is: sum_c U3[i,c] * V3[j,c] * W3[k,c] = 1 for all i,j,k
    # We need: sum_c U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c] = 1
    # 
    # If we set U_full[i*m+j, c] = U3[i, c] * alpha_j, V_full[j*p+k, c] = V3[j, c] * beta_k,
    # W_full[k*n+i, c] = W3[k, c] * gamma_i, with alpha_j*beta_k*gamma_i = 1 for all i,j,k,
    # then the product is the same.
    #
    # Simplest: U_full[i*m+j, c] = U3[i, c], V_full[j*p+k, c] = V3[j, c], W_full[k*n+i, c] = W3[k, c]
    # Then sum_c U3[i,c]*V3[j,c]*W3[k,c] which is exactly what we want!
    
    U_full = np.zeros((n * m, rank))
    V_full = np.zeros((m * p, rank))
    W_full = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            U_full[i * m + j, :] = U3[i, :]
    for j in range(m):
        for k in range(p):
            V_full[j * p + k, :] = V3[j, :]
    for k in range(p):
        for i in range(n):
            W_full[k * n + i, :] = W3[k, :]
    
    return U_full, V_full, W_full, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try ranks from low to high
    for rank in [12, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]:
        if time.time() > DEADLINE:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(50):
            if time.time() > DEADLINE:
                break
            
            result = _als_decompose_full(n, m, p, rank, max_iter=50, tol=1e-8, seed=seed)
            if result is not None:
                U_full, V_full, W_full, r = result
                if _verify(U_full, V_full, W_full, n, m, p):
                    # Convert to lists
                    U_list = U_full.tolist()
                    V_list = V_full.tolist()
                    W_list = W_full.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        if time.time() > DEADLINE:
            break
    
    # Fall back to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
