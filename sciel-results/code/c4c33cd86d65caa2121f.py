"""Matrix multiplication 2x3 x 3x5 rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor T[i,j,k] = delta[i*m+j, k*n+i] structure.
    Actually T is a 3-tensor of shape (n*m, m*p, p*n) where
    T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    n, m, p = T.shape[0], T.shape[1] // 3, T.shape[2] // 2
    # Actually need to be careful with shapes
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    n, m, p = 2, 3, 5
    rank = U.shape[1]
    # Reshape for efficient computation
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    # Compute tensor and compare
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                val = np.dot(U_mat[i], V_mat[j] * W_mat[k])
                if abs(val - T[i, j, k]) > tol:
                    return False
    return True

def _try_rank_search(T, max_rank, deadline, seed=42):
    """Search for a valid decomposition using iterative methods."""
    n, m, p = 2, 3, 5
    dim1, dim2, dim3 = n * m, m * p, p * n  # 6, 15, 10
    rng = np.random.RandomState(seed)
    
    best_result = None
    best_residual = float('inf')
    
    for target_rank in range(max_rank, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random sparse patterns
            U = np.zeros((dim1, target_rank))
            V = np.zeros((dim2, target_rank))
            W = np.zeros((dim3, target_rank))
            
            # Set up initial support: each tensor entry should be covered
            # Use a greedy approach to assign supports
            entries = [(i, j, k) for i in range(dim1) for j in range(dim2) for k in range(dim3) if T[i,j,k] != 0]
            
            # Assign each entry to a random rank component
            for (i, j, k) in entries:
                r = rng.randint(target_rank)
                # Distribute the value 1 among the three factors
                # Simple: put 1 in one, 1 in another, 1 in third
                U[i, r] = 1
                V[j, r] = 1
                W[k, r] = 1
            
            # Now iterate to refine
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Compute residual for each factor
                # Update U: for each (i, r), U[i,r] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r] / (sum_jk V[j,r]^2 * W[k,r]^2 + eps)
                # This is a simplified version of tensor ring approximation
                
                # Actually, let's use a different approach: alternating least squares
                # For fixed V, W, solve for U:
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For each i: T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])
                # So T[i,:,:] = U[i,:] @ diag(V*W) ... not quite
                # Better: reshape
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed i: T[i,:, :] = sum_r U[i,r] * outer(V[:,r], W[:,r])
                # So T[i].reshape(dim2*dim3) = U[i,:] @ (V*W).reshape(dim2*dim3, -1)
                
                # Compute VW tensor: (dim2, dim3, target_rank)
                VW = np.einsum('jr,kr->jkr', V, W)
                VW_mat = VW.reshape(dim2 * dim3, target_rank)
                
                # Solve for U: T[i].reshape(-1) = U[i,:] @ VW_mat
                # U[i,:] = T[i].reshape(-1) @ pinv(VW_mat)
                try:
                    U_new = np.einsum('ijk,kr->ir', T, V * W)  # This isn't right
                except:
                    pass
                
                # Let me do it properly with least squares
                # For each i, solve min_U sum_{j,k} (T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
                # This is: T[i].reshape(-1) = VW_mat.T @ U[i,:].T  =>  U[i,:] = (VW_mat @ VW_mat.T)^-1 @ VW_mat @ T[i].reshape(-1)
                
                # Compute VW_mat @ VW_mat.T (target_rank x target_rank)
                A = VW_mat.T @ VW_mat  # (target_rank, target_rank)
                b_all = np.einsum('ijk,jk->ik', T, VW_mat.T)  # This is wrong too
                
                # Let me be more careful
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # Define M_r = outer(V[:,r], W[:,r])  (dim2 x dim3)
                # Then T[i] = sum_r U[i,r] * M_r
                # So T[i].reshape(-1) = U[i,:] @ M_all  where M_all is (target_rank, dim2*dim3)
                # M_all[r] = M_r.reshape(-1)
                
                M_all = (V[:, :, None] * W[:, None, :]).transpose(2, 0, 1).reshape(target_rank, dim2 * dim3)
                
                # Solve U[i,:] @ M_all = T[i].reshape(-1)
                # U[i,:] = T[i].reshape(-1) @ pinv(M_all)
                M_pinv = np.linalg.pinv(M_all)
                U_new = np.einsum('ijk,kl->il', T, M_pinv)
                
                # Similarly for V and W
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed j: T[:,j,:] = sum_r V[j,r] * outer(U[:,r], W[:,r])
                UW_all = (U[:, :, None] * W[:, None, :]).transpose(2, 0, 1).reshape(target_rank, dim1 * dim3)
                UW_pinv = np.linalg.pinv(UW_all)
                V_new = np.einsum('ijk,kl->jl', T, UW_pinv)
                
                # For fixed k: T[:,:,k] = sum_r W[k,r] * outer(U[:,r], V[:,r])
                UV_all = (U[:, :, None] * V[:, None, :]).transpose(2, 0, 1).reshape(target_rank, dim1 * dim2)
                UV_pinv = np.linalg.pinv(UV_all)
                W_new = np.einsum('ijk,kl->kl', T, UV_pinv)
                
                U = U_new
                V = V_new
                W = W_new
                
                # Check residual
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-12)
                
                if residual < best_residual:
                    best_residual = residual
                    if residual < 1e-8:
                        # Found a valid decomposition!
                        # Try to make entries rational
                        U_frac = np.array([[Fraction(x).limit_denominator(1000) for x in row] for row in U])
                        V_frac = np.array([[Fraction(x).limit_denominator(1000) for x in row] for row in V])
                        W_frac = np.array([[Fraction(x).limit_denominator(1000) for x in row] for row in W])
                        
                        # Convert to float for verification
                        U_check = np.array([[float(f) for f in row] for row in U_frac])
                        V_check = np.array([[float(f) for f in row] for row in V_frac])
                        W_check = np.array([[float(f) for f in row] for row in W_frac])
                        
                        if _check_decomposition(U_check, V_check, W_check, T, tol=1e-6):
                            return {"U": U_frac.tolist(), "V": V_frac.tolist(), "W": W_frac.tolist(), "rank": target_rank}
                        
                        # Even if rational check fails, return the float version if residual is tiny
                        if residual < 1e-10:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": target_rank}
                
                if residual < 1e-10:
                    break
            
            # After iterations, check if we have a good solution
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-12)
            if residual < 1e-8:
                U_frac = np.array([[Fraction(x).limit_denominator(10000) for x in row] for row in U])
                V_frac = np.array([[Fraction(x).limit_denominator(10000) for x in row] for row in V])
                W_frac = np.array([[Fraction(x).limit_denominator(10000) for x in row] for row in W])
                U_check = np.array([[float(f) for f in row] for row in U_frac])
                V_check = np.array([[float(f) for f in row] for row in V_frac])
                W_check = np.array([[float(f) for f in row] for row in W_frac])
                if _check_decomposition(U_check, V_check, W_check, T, tol=1e-6):
                    return {"U": U_frac.tolist(), "V": V_frac.tolist(), "W": W_frac.tolist(), "rank": target_rank}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s, leave margin
    
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank
    # Based on known results, try ranks from 25 down to 20
    for target_rank in [25, 24, 23, 22, 21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_search(T, target_rank, deadline, seed=target_rank * 100 + 42)
        if result is not None:
            r = result["rank"]
            if r < best_rank:
                best_rank = r
                best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    return best_result
