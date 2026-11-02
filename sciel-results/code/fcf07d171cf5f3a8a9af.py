import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def _build_3x3x2_tensor():
    """Build the 3x3x2 matrix multiplication tensor."""
    n, m, p = 3, 3, 2
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _optimize_rank_decomposition(T, target_rank, seed=42, max_iter=500, tol=1e-10):
    """
    Find a rank-`target_rank` decomposition of tensor T.
    T has shape (a, b, c).
    Returns (U, V, W) or None if failed.
    """
    a, b, c = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(a, target_rank) * 0.5
    V = rng.randn(b, target_rank) * 0.5
    W = rng.randn(c, target_rank) * 0.5
    
    def loss(params):
        U_p = params[:a*target_rank].reshape(a, target_rank)
        V_p = params[a*target_rank:a*target_rank + b*target_rank].reshape(b, target_rank)
        W_p = params[a*target_rank + b*target_rank:].reshape(c, target_rank)
        recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p, optimize=True)
        return np.sum((recon - T)**2)
    
    def loss_grad(params):
        U_p = params[:a*target_rank].reshape(a, target_rank)
        V_p = params[a*target_rank:a*target_rank + b*target_rank].reshape(b, target_rank)
        W_p = params[a*target_rank + b*target_rank:].reshape(c, target_rank)
        
        recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p, optimize=True)
        residual = recon - T
        
        # Gradient w.r.t. U
        grad_U = np.einsum('abc,br,cr->ar', residual, V_p, W_p, optimize=True)
        grad_V = np.einsum('abc,ar,cr->br', residual, U_p, W_p, optimize=True)
        grad_W = np.einsum('abc,ar,br->cr', residual, U_p, V_p, optimize=True)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_params = params0.copy()
    best_loss = loss(params0)
    
    for restart in range(5):
        if restart > 0:
            params0 = rng.randn(len(params0)) * 0.5
        
        result = minimize(loss, params0, jac=loss_grad, method='BFGS', 
                         options={'maxiter': max_iter, 'gtol': 1e-8})
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
    
    U_final = best_params[:a*target_rank].reshape(a, target_rank)
    V_final = best_params[a*target_rank:a*target_rank + b*target_rank].reshape(b, target_rank)
    W_final = best_params[a*target_rank + b*target_rank:].reshape(c, target_rank)
    
    recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final, optimize=True)
    if np.max(np.abs(recon - T)) < tol:
        return U_final, V_final, W_final
    
    return None

def _build_3x3x6_from_3x3x2():
    """
    Build 3x3x6 decomposition from three 3x3x2 decompositions.
    3x3x6 = 3x3x2 ⊕ 3x3x2 ⊕ 3x3x2
    """
    n, m, p = 3, 3, 6
    
    # For 3x3x6, we have k in 0..5
    # Split into three groups: k=0,1; k=2,3; k=4,5
    # For each group, the 3x3x2 tensor is the same
    
    T_3x3x2 = _build_3x3x2_tensor()
    
    # Try to find rank-13 decomposition for 3x3x2
    # First try rank 13, then 12, 11, ...
    best_decomp = None
    for target_rank in [13, 12, 11, 10, 9]:
        decomp = _optimize_rank_decomposition(T_3x3x2, target_rank, seed=42)
        if decomp is not None:
            best_decomp = decomp
            break
    
    if best_decomp is None:
        # Fallback: use schoolbook for 3x3x2 (rank 18)
        U_3x3x2 = np.zeros((9, 18))
        V_3x3x2 = np.zeros((6, 18))
        W_3x3x2 = np.zeros((6, 18))
        r = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U_3x3x2[i*3+j, r] = 1.0
                    V_3x3x2[j*2+k, r] = 1.0
                    W_3x3x2[k*3+i, r] = 1.0
                    r += 1
        best_decomp = (U_3x3x2, V_3x3x2, W_3x3x2)
    
    U_block, V_block, W_block = best_decomp
    block_rank = U_block.shape[1]
    
    # Now combine three blocks
    total_rank = 3 * block_rank
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((18, total_rank))
    W_full = np.zeros((18, total_rank))
    
    for block in range(3):
        # Block b handles k = 2*b, 2*b+1
        r_offset = block * block_rank
        
        # U is the same for all blocks (A matrix is 3x3)
        U_full[:, r_offset:r_offset+block_rank] = U_block
        
        # V: for block b, V rows correspond to (j, k) where k = 2*b + k_local
        for j in range(3):
            for k_local in range(2):
                k_global = 2 * block + k_local
                v_row = j * 6 + k_global  # Wait, V has shape (m*p, rank) = (3*6, rank) = (18, rank)
                # V is indexed by (j, k) where j is row of B (0-2), k is col of B (0-5)
                # So V row index = j * p + k = j * 6 + k
                V_full[v_row, r_offset:r_offset+block_rank] = V_block[j * 2 + k_local]
        
        # W: W is indexed by (k, i) where k is col of B (0-5), i is row of output (0-2)
        # W row index = k * n + i = k * 3 + i
        for k_local in range(2):
            k_global = 2 * block + k_local
            for i in range(3):
                w_row = k_global * 3 + i
                W_full[w_row, r_offset:r_offset+block_rank] = W_block[k_local * 3 + i]
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim, b_dim, c_dim = n*m, m*p, p*n
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    T = np.zeros((a_dim, b_dim, c_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    max_err = np.max(np.abs(recon - T))
    return max_err < tol, max_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s safety margin
    
    # Try the block decomposition approach
    try:
        U, V, W = _build_3x3x6_from_3x3x2()
        valid, err = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            rank = U.shape[1]
            if rank < n * m * p:  # Better than schoolbook
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback: schoolbook
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
