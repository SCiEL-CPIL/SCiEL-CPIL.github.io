"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor T[i,j,k] = 1 if output (i,k) uses input (i,j) and (j,k)
    # T is a 3D tensor of shape (n, m, p)
    # Matricization: U has shape (nm, rank), V (mp, rank), W (pn, rank)
    # The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build target tensor in matricized form
    # T_mat[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # Actually, the tensor is T[i,j,k] = 1, and the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So the target is: for each (i,j,k), T[i,j,k] = 1
    # This means we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook: rank = n*m*p
    schoolbook_rank = n * m * p
    
    # Try to find a decomposition with lower rank
    # Use randomized search: initialize with random rank-r factors, then optimize
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check all (i,j,k) entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=2000, seed=None):
        """Try to find a rank-r decomposition using iterative optimization."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        nm, mp, pn = n*m, m*p, p*n
        
        # Initialize with small random values
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Use a simple gradient-free approach: coordinate descent on each factor
        # The objective is to minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        
        # Precompute which (i,j,k) tuples are valid (all are valid for full matrix mult)
        # All n*m*p entries should equal 1
        
        # Use alternating least squares-like approach
        # Fix two factors, solve for the third
        
        best_loss = float('inf')
        best_factors = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iters):
            # Solve for U given V, W
            # For each row i*m+j of U, we need: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # This is a linear system: for fixed (i,j), let a_r = V[j*p+k,r] * W[k*n+i,r] for each k
            # We need U[i*m+j, :] @ a_k = 1 for all k
            # So: A @ U[i*m+j, :]^T = 1, where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    # Build A matrix of shape (p, r)
                    A = np.zeros((p, r))
                    for k in range(p):
                        for r_idx in range(r):
                            A[k, r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    # Solve A @ x = 1
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k), need sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
                    A = np.zeros((n, r))
                    for i in range(n):
                        for r_idx in range(r):
                            A[i, r_idx] = U[i*m+j, r_idx] * W[k*n+i, r_idx]
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Solve for W given U, V
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i), need sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
                    A = np.zeros((m, r))
                    for j in range(m):
                        for r_idx in range(r):
                            A[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        loss += (val - 1.0) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_factors = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-12:
                break
        
        if best_loss < 1e-9:
            return best_factors, True
        return best_factors, False
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: at least max(nm, mp, pn) / something, but practically try from reasonable values
    # For 3x4x4, schoolbook is 48. Try from 30 up to 48.
    
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try lower ranks
    for r in range(35, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            factors, success = try_rank(r, max_iters=500, seed=42 + r * 100 + attempt)
            if success:
                U, V, W = factors
                # Convert to list of lists with rounded values
                U_list = [[round(U[i, j], 6) for j in range(r)] for i in range(nm)]
                V_list = [[round(V[i, j], 6) for j in range(r)] for i in range(mp)]
                W_list = [[round(W[i, j], 6) for j in range(r)] for i in range(pn)]
                
                # Verify with the rounded values
                if verify(U_list, V_list, W_list, tol=1e-6):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a valid decomposition with rank r, try even lower
            continue
        else:
            # If we can't find rank r, try r+1 (already in loop)
            pass
    
    return best_result
# EVOLVE-BLOCK-END
