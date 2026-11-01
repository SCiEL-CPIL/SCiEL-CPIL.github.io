"""
Tensor decomposition for 2x3 x 3x7 matrix multiplication with rank minimization.

Uses a systematic approach:
1. Start with the schoolbook rank-42 decomposition.
2. Use linear algebra to check if we can reduce rank by finding linear dependencies
   among the rank-1 terms when projected onto subspaces.
3. Try structured low-rank approximations using known matrix multiplication techniques.
4. Use iterative refinement to find a valid low-rank decomposition.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE = 145.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def remaining():
    return DEADLINE - elapsed()


def schoolbook_decomposition():
    """Generate the trivial schoolbook decomposition of rank 42."""
    n, m, p = 2, 3, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def build_target_tensor():
    """Build the standard matrix multiplication tensor for 2x3 * 3x7."""
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i_A in range(n):
        for j in range(m):
            for k in range(p):
                i_input = i_A * m + j
                j_input = j * p + k
                k_output = k * n + i_A
                T[i_input, j_input, k_output] = 1.0
    return T


def verify_decomposition(U, V, W, T=None, tol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    Returns (is_valid, max_error).
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.ndim == 1:
        U_arr = U_arr.reshape(nm, -1)
    if V_arr.ndim == 1:
        V_arr = V_arr.reshape(mp, -1)
    if W_arr.ndim == 1:
        W_arr = W_arr.reshape(pn, -1)
    
    rank = U_arr.shape[1]
    
    if T is None:
        T = build_target_tensor()
    
    # Compute the tensor: T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    T_computed = np.zeros((nm, mp, pn), dtype=float)
    for t in range(rank):
        T_computed += np.einsum('a,b,c->abc', U_arr[:, t], V_arr[:, t], W_arr[:, t])
    
    max_err = np.max(np.abs(T_computed - T))
    return max_err < tol, max_err


def try_rank_reduction_by_projection(T, U, V, W, target_rank, max_iters=500):
    """
    Try to reduce the rank by finding a low-rank representation.
    Uses iterative projection method.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = U_arr.shape[1]
    
    if current_rank <= target_rank:
        return None
    
    # Try to find a rank-reduced decomposition
    # Strategy: randomly select target_rank columns and try to adjust them
    # to still represent the tensor exactly.
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(max_iters):
        if remaining() < 5:
            break
        
        # Randomly select target_rank indices from the current rank
        idx = np.random.choice(current_rank, target_rank, replace=False)
        
        # Get the sub-matrices
        U_sub = U_arr[:, idx].copy()
        V_sub = V_arr[:, idx].copy()
        W_sub = W_arr[:, idx].copy()
        
        # Try to find adjustments to make this work
        # We need: sum_t U_sub[:,t] * V_sub[:,t] * W_sub[:,t] = T
        
        # This is a hard problem. Let's try a different approach:
        # Use the fact that the tensor is sparse and structured.
        
        # Alternative: try to find a rank-r decomposition from scratch
        # using random initialization and gradient descent
        U_rand = np.random.randn(nm, target_rank) * 0.5
        V_rand = np.random.randn(mp, target_rank) * 0.5
        W_rand = np.random.randn(pn, target_rank) * 0.5
        
        # Normalize
        for t in range(target_rank):
            norm = np.sqrt(np.sum(U_rand[:, t]**2) * np.sum(V_rand[:, t]**2) * np.sum(W_rand[:, t]**2))
            if norm > 1e-10:
                scale = 1.0 / (norm ** (1.0/3.0))
                U_rand[:, t] *= scale
                V_rand[:, t] *= scale
                W_rand[:, t] *= scale
        
        # Iterate: update one factor at a time
        for it in range(200):
            if remaining() < 5:
                break
            
            # Update U
            # For each a, U[a,t] should satisfy:
            # sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
            # Fix V and W, solve for U
            # T[a,:,:] = sum_t U[a,t] * (V[:,t] * W[:,t]).T flattened
            # This is a linear system for U[a,:]
            
            # Build the linear system for U
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # Let M[b,c,t] = V[b,t] * W[c,t]
            # Then T[a] = U[a] @ M.reshape(mp*pn, -1).T
            # where T[a] is T[a,:,:].flatten()
            
            M = np.einsum('bt,ct->bct', V_rand, W_rand)  # (mp, pn, target_rank)
            M_flat = M.transpose(2, 0, 1).reshape(target_rank, mp * pn)  # (target_rank, mp*pn)
            
            # Solve U[a] @ M_flat.T = T[a].flatten()
            # i.e., M_flat @ U[a].T = T[a].flatten()
            # Least squares: U[a].T = pinv(M_flat) @ T[a].flatten()
            
            T_flat = T.reshape(nm, mp * pn)  # (nm, mp*pn)
            
            # Solve the linear system
            try:
                # U[a] @ M_flat.T = T[a]
                # M_flat @ U[a].T = T[a]
                # U[a].T = pinv(M_flat) @ T[a]
                pinv_M = np.linalg.pinv(M_flat)
                U_new = (pinv_M @ T_flat.T).T  # (nm, target_rank)
            except:
                break
            
            # Update V
            M2 = np.einsum('at,ct->act', U_new, W_rand)  # (nm, pn, target_rank)
            M2_flat = M2.transpose(2, 0, 1).reshape(target_rank, nm * pn)
            T2 = T.reshape(mp, nm * pn)  # Need to reshape T appropriately
            
            # T[b,c,a] = sum_t V[b,t] * U[a,t] * W[c,t]
            # T2[b] = V[b] @ M2_flat.T
            T2_flat = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                pinv_M2 = np.linalg.pinv(M2_flat)
                V_new = (pinv_M2 @ T2_flat.T).T
            except:
                break
            
            # Update W
            M3 = np.einsum('at,bt->abt', U_new, V_new)  # (nm, mp, target_rank)
            M3_flat = M3.transpose(2, 0, 1).reshape(target_rank, nm * mp)
            T3_flat = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                pinv_M3 = np.linalg.pinv(M3_flat)
                W_new = (pinv_M3 @ T3_flat.T).T
            except:
                break
            
            # Check convergence
            T_computed = np.zeros((nm, mp, pn), dtype=float)
            for t in range(target_rank):
                T_computed += np.einsum('a,b,c->abc', U_new[:, t], V_new[:, t], W_new[:, t])
            
            err = np.max(np.abs(T_computed - T))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            if err < 1e-9:
                break
            
            U_new, V_new, W_new = U_new, V_new, W_new
        
        if best_err < 1e-9:
            break
    
    if best_U is not None and best_err < 1e-9:
        return best_U, best_V, best_W
    
    return None


def try_structured_decomposition(T):
    """
    Try to find a structured low-rank decomposition using
    the structure of the matrix multiplication problem.
    
    For 2x3 * 3x7, we can exploit the small dimensions.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # The tensor has a specific structure:
    # T[i*m+j, j*p+k, k*n+i] = 1 for i in [0,n), j in [0,m), k in [0,p)
    
    # One approach: group by the middle index j
    # For each j, we have n*p = 14 entries
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # This means for fixed j, the sub-tensor is:
    # T_j[i, k] = 1 (for all i, k) but with specific index mappings
    
    # Actually, let's think about this differently.
    # The matrix multiplication can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can try to use the fact that n=2 is small.
    # For each pair (j,k), we need to compute:
    # C[0,k] += A[0,j] * B[j,k]
    # C[1,k] += A[1,j] * B[j,k]
    
    # This is like multiplying a 2-vector by a scalar for each (j,k).
    # The rank of this operation is at most 2 (for the 2-vector).
    
    # But we need to do this for all j,k pairs, and the A and B values
    # are independent, so we can't easily share rank-1 terms.
    
    # Let's try a different approach: use the Strassen-like idea
    # to reduce the number of multiplications.
    
    # For 2x3 * 3x7, we can potentially use block decomposition.
    # But the dimensions are quite different, so this may not help much.
    
    # Let's just try random search with the iterative method
    # for various target ranks.
    
    return None


def construct():
    """
    Construct a tensor decomposition for 2x3 x 3x7 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    best_result = schoolbook_decomposition()
    T = build_target_tensor()
    
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Try to find a lower-rank decomposition using iterative methods
    # Start from higher ranks and work down
    
    found = False
    for target_rank in range(30, 15, -1):
        if remaining() < 10:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if remaining() < 5:
                break
            
            U_rand = np.random.randn(nm, target_rank) * 0.5
            V_rand = np.random.randn(mp, target_rank) * 0.5
            W_rand = np.random.randn(pn, target_rank) * 0.5
            
            # Normalize each rank-1 term
            for t in range(target_rank):
                norm = np.sqrt(np.sum(U_rand[:, t]**2) * np.sum(V_rand[:, t]**2) * np.sum(W_rand[:, t]**2))
                if norm > 1e-10:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U_rand[:, t] *= scale
                    V_rand[:, t] *= scale
                    W_rand[:, t] *= scale
            
            # Iterate
            U_new, V_new, W_new = U_rand.copy(), V_rand.copy(), W_rand.copy()
            converged = False
            
            for it in range(300):
                if remaining() < 5:
                    break
                
                # Update U
                M = np.einsum('bt,ct->bct', V_new, W_new)
                M_flat = M.transpose(2, 0, 1).reshape(target_rank, mp * pn)
                T_flat = T.reshape(nm, mp * pn)
                try:
                    pinv_M = np.linalg.pinv(M_flat)
                    U_new = (pinv_M @ T_flat.T).T
                except:
                    break
                
                # Update V
                M2 = np.einsum('at,ct->act', U_new, W_new)
                M2_flat = M2.transpose(2, 0, 1).reshape(target_rank, nm * pn)
                T2_flat = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    pinv_M2 = np.linalg.pinv(M2_flat)
                    V_new = (pinv_M2 @ T2_flat.T).T
                except:
                    break
                
                # Update W
                M3 = np.einsum('at,bt->abt', U_new, V_new)
                M3_flat = M3.transpose(2, 0, 1).reshape(target_rank, nm * mp)
                T3_flat = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    pinv_M3 = np.linalg.pinv(M3_flat)
                    W_new = (pinv_M3 @ T3_flat.T).T
                except:
                    break
                
                # Check convergence
                T_computed = np.zeros((nm, mp, pn), dtype=float)
                for t in range(target_rank):
                    T_computed += np.einsum('a,b,c->abc', U_new[:, t], V_new[:, t], W_new[:, t])
                
                err = np.max(np.abs(T_computed - T))
                if err < 1e-9:
                    converged = True
                    break
            
            if converged:
                # Found a valid decomposition!
                best_result = {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    # Verify the result
    valid, max_err = verify_decomposition(best_result["U"], best_result["V"], best_result["W"], T)
    if not valid:
        # Fallback to schoolbook
        best_result = schoolbook_decomposition()
    
    return best_result
