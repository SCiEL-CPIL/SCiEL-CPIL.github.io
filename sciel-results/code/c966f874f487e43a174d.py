"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U @ V @ W^T reconstructs the multiplication tensor exactly."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1 if output index (i,k) gets A[i,j]*B[j,k]
    # T is indexed by (i, j, k) where A is n x m, B is m x p
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check by reconstructing T and comparing to the true tensor
    
    # True tensor: T[i,j,k] = 1 for all i,j,k (it's the identity-like multiplication tensor)
    # Actually the multiplication tensor has T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T_{i,j,k} = 1 for all valid i,j,k
    
    # Reconstruct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can compute this efficiently:
    # For each c: outer product contribution
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U_arr[:, c]  # length n*m
        vc = V_arr[:, c]  # length m*p
        wc = W_arr[:, c]  # length p*n
        
        uc_reshaped = uc.reshape(n, m)
        vc_reshaped = vc.reshape(m, p)
        wc_reshaped = wc.reshape(p, n)
        
        # T[i,j,k] += uc_reshaped[i,j] * vc_reshaped[j,k] * wc_reshaped[k,i]
        # This is a bit tricky. Let's think of it differently.
        # For fixed c, the contribution to T is:
        # T_c[i,j,k] = uc_reshaped[i,j] * vc_reshaped[j,k] * wc_reshaped[k,i]
        
        # We can compute this as:
        # For each i: uc_reshaped[i,:] * (for each j: vc_reshaped[j,:] * for each k: wc_reshaped[k,i])
        # Actually let's just do it directly with broadcasting
        # T_c[i,j,k] = uc_reshaped[i,j] * vc_reshaped[j,k] * wc_reshaped[k,i]
        
        # Build a 3D array
        # uc_reshaped[i,j] -> shape (n, m, 1)
        # vc_reshaped[j,k] -> shape (1, m, p)  
        # wc_reshaped[k,i] -> shape (p, 1, n)
        
        contrib = (uc_reshaped[:, :, np.newaxis] * 
                   vc_reshaped[np.newaxis, :, :] * 
                   wc_reshaped[:, np.newaxis, :])
        # contrib has shape (n, m, p) where contrib[i,j,k] = uc[i,j]*vc[j,k]*wc[k,i]
        T += contrib
    
    # True tensor is all ones
    true_T = np.ones((n, m, p), dtype=float)
    return np.max(np.abs(T - true_T)) < tol

def _make_integer_or_half(U, V, W, tol=1e-6):
    """Try to round entries to nearest integer or half-integer."""
    U_rounded = np.round(np.array(U, dtype=float) * 2) / 2.0
    V_rounded = np.round(np.array(V, dtype=float) * 2) / 2.0
    W_rounded = np.round(np.array(W, dtype=float) * 2) / 2.0
    return U_rounded, V_rounded, W_rounded

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find lower rank decompositions
    # Strategy: random initialization + iterative refinement
    # We search for rank r < best_rank
    
    np.random.seed(42)
    
    # Try various ranks from low to high, stopping when we find a valid one
    # Theoretical lower bound for 3x5x5 matrix mult is not trivially known,
    # but let's try ranks from some reasonable minimum up
    
    # For a 3x5x5 multiplication, the tensor has 3*5*5 = 75 entries all = 1
    # A rank-r decomposition requires r terms
    # Let's try to find the minimum r
    
    # Use a random tensor decomposition approach:
    # Initialize U, V, W randomly, then iteratively refine
    
    def try_find_decomposition(target_rank, max_iters=2000, tol=1e-8):
        """Try to find a rank-target_rank decomposition."""
        n, m, p = DIMENSIONS
        U_size = n * m
        V_size = m * p
        W_size = p * n
        
        # True tensor
        true_T = np.ones((n, m, p))
        
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(5):
            # Random initialization
            U = np.random.randn(U_size, target_rank)
            V = np.random.randn(V_size, target_rank)
            W = np.random.randn(W_size, target_rank)
            
            # Iterative refinement: alternate fixing two matrices and solving for the third
            for it in range(max_iters):
                if time.time() > deadline:
                    return None, None, None, best_err, best_U, best_V, best_W
                
                # Fix V and W, solve for U
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed V, W: T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                # Let A_c[i*m+j, j*p+k, k*n+i]... this is complex
                # 
                # Alternative: view as linear system
                # For each (i,j,k): T[i,j,k] = sum_c U[i*m+j,c] * B_c[i,j,k]
                # where B_c[i,j,k] = V[j*p+k,c] * W[k*n+i,c]
                # 
                # This is a linear system in U. Let's set up the matrix.
                # Number of equations: n*m*p = 75
                # Number of unknowns: U_size * target_rank = 15 * target_rank
                # 
                # Let's build the system:
                # For each (i,j,k), the equation is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # 
                # This is linear in U. Let's form the matrix M where:
                # M[eq_idx, (i*m+j)*target_rank + c] = V[j*p+k, c] * W[k*n+i, c]
                # and rhs[eq_idx] = 1
                
                eq_count = n * m * p
                unknown_count = U_size * target_rank
                
                M = np.zeros((eq_count, unknown_count))
                rhs = np.ones(eq_count)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                M[eq_idx, (i * m + j) * target_rank + c] = V[j * p + k, c] * W[k * n + i, c]
                            eq_idx += 1
                
                # Solve least squares
                try:
                    U_sol, residuals, rank_sys, sv = np.linalg.lstsq(M, rhs, rcond=tol)
                    U = U_sol.reshape(U_size, target_rank)
                except:
                    continue
                
                # Fix U and W, solve for V
                M = np.zeros((eq_count, V_size * target_rank))
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                M[eq_idx, (j * p + k) * target_rank + c] = U[i * m + j, c] * W[k * n + i, c]
                            eq_idx += 1
                try:
                    V_sol, residuals, rank_sys, sv = np.linalg.lstsq(M, rhs, rcond=tol)
                    V = V_sol.reshape(V_size, target_rank)
                except:
                    continue
                
                # Fix U and V, solve for W
                M = np.zeros((eq_count, W_size * target_rank))
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                M[eq_idx, (k * n + i) * target_rank + c] = U[i * m + j, c] * V[j * p + k, c]
                            eq_idx += 1
                try:
                    W_sol, residuals, rank_sys, sv = np.linalg.lstsq(M, rhs, rcond=tol)
                    W = W_sol.reshape(W_size, target_rank)
                except:
                    continue
                
                # Check error
                U_arr = U
                V_arr = V
                W_arr = W
                T = np.zeros((n, m, p), dtype=float)
                for c in range(target_rank):
                    uc = U_arr[:, c].reshape(n, m)
                    vc = V_arr[:, c].reshape(m, p)
                    wc = W_arr[:, c].reshape(p, n)
                    contrib = (uc[:, :, np.newaxis] * 
                               vc[np.newaxis, :, :] * 
                               wc[:, np.newaxis, :])
                    T += contrib
                
                err = np.max(np.abs(T - true_T))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < tol:
                    return U, V, W, err, None, None, None
        
        return best_U, best_V, best_W, best_err, None, None, None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # The minimum rank for 3x5x5 matrix multiplication is unknown but likely < 75
    # Let's try from low ranks upward
    
    # Actually, let's be smarter. Let's try a range of ranks.
    # For small matrix multiplications, the rank is often close to n*m*p but can be lower.
    # Let's try ranks from 30 to 74 (below schoolbook)
    
    # But first, let's try a more systematic approach with fewer iterations per rank
    # to stay within time budget
    
    # Try a few promising ranks
    ranks_to_try = list(range(40, 75)) + list(range(30, 40))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        U_found, V_found, W_found, err, _, _, _ = try_find_decomposition(target_rank, max_iters=500)
        
        if U_found is not None and err < 1e-8:
            # Verify
            if _verify(U_found, V_found, W_found, n, m, p):
                # Try to make entries integer/half-integer
                U_r, V_r, W_r = _make_integer_or_half(U_found, V_found, W_found)
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                else:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_found, V_found, W_found
                break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < rank_schoolbook:
        return {
            "U": best_U.tolist() if isinstance(best_U, np.ndarray) else best_U,
            "V": best_V.tolist() if isinstance(best_V, np.ndarray) else best_V,
            "W": best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
