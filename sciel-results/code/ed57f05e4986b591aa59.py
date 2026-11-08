"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Trivial solution with rank = n*m*p = 42
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    # Compute lower bounds
    T0 = tensor.reshape(n*m, -1)
    rank0 = np.linalg.matrix_rank(T0)
    T1 = tensor.transpose(1,0,2).reshape(m*p, -1)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = tensor.transpose(2,0,1).reshape(p*n, -1)
    rank2 = np.linalg.matrix_rank(T2)
    lower_bound = max(rank0, rank1, rank2)
    
    def verify_decomposition(U, V, W, tensor, tol=1e-10):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = tensor"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, tensor, atol=tol)
    
    def try_rank(trial_rank, max_time=10.0):
        """Try to find a decomposition of given rank using structured approach"""
        start_time = time.time()
        
        # Strategy: Use the structure of matrix multiplication
        # For each (i,j,k) triple, the tensor has a 1 at position (i*m+j, j*p+k, k*n+i)
        # We can group these by the middle index j
        
        # Try different structured patterns for U
        # Pattern: U[i*m+j, r] is nonzero only for certain (i,j) pairs
        
        np.random.seed(42 + trial_rank)
        
        # Try multiple random initializations with structure
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            if time.time() > deadline - 15:
                break
            
            # Create structured U: each row (i,j) gets 1-2 nonzero entries
            U = np.zeros((n*m, trial_rank))
            for i in range(n):
                for j in range(m):
                    a = i*m + j
                    # Choose 1-2 columns for this row
                    num_nz = np.random.randint(1, 3)
                    cols = np.random.choice(trial_rank, size=num_nz, replace=False)
                    for c in cols:
                        U[a, c] = np.random.choice([-1, 1])
            
            # Check if U has full row rank
            if np.any(np.all(U == 0, axis=1)):
                continue
            
            # Now solve for V and W using ALS
            UtU = U.T @ U
            
            # Initialize V and W
            V = np.random.randn(m*p, trial_rank) * 0.1
            W = np.random.randn(p*n, trial_rank) * 0.1
            
            converged = False
            for iteration in range(100):
                if time.time() - start_time > max_time:
                    break
                if time.time() > deadline - 15:
                    break
                
                # Update W: for each c, W[:,c] = (sum_{a,b} T[a,b,c] U[a,:] V[b,:]) / (sum_{a,b} T[a,b,c] U[a,r] V[b,r])
                # More precisely: W[:,c] = argmin_w sum_{a,b} T[a,b,c] (U[a,r] V[b,r] w_r - T[a,b,c])^2
                # This gives: W[:,c] = (U^T diag(T[:, :, c]) V) / (U^T diag(T[:, :, c]) V * 1) ... 
                # Actually: for each c, the contribution is sum_r U[:,r] V[:,r] W[c,r]
                # We want: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all a,b,c
                # For fixed U,V: W[c,r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r] / sum_{a,b} (U[a,r] V[b,r])^2
                # But this is not quite right. Let me use the proper ALS update.
                
                # Proper ALS: minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
                # W update: for each c, W[:,c] = (sum_{a,b} T[a,b,c] U[a,:] V[b,:]) / (U^T U * V^T V) element-wise? No.
                # The gradient with respect to W[c,r] is: 2 * sum_{a,b} T[a,b,c] U[a,r] V[b,r] - 2 * sum_{a,b} (sum_{r'} U[a,r'] V[b,r'] W[c,r']) U[a,r] V[b,r]
                # Setting to 0: sum_{a,b} T[a,b,c] U[a,r] V[b,r] = sum_{a,b} (sum_{r'} U[a,r'] V[b,r'] W[c,r']) U[a,r] V[b,r]
                # This is: (U^T T_c V) = (U^T (T_c * (U V^T)) V) where T_c is the c-th slice
                # Actually: rhs_W[c,r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
                # lhs_W[r,r'] = sum_{a,b} U[a,r] U[a,r'] V[b,r] V[b,r'] = (U^T U)[r,r'] * (V^T V)[r,r']
                
                VtV = V.T @ V
                G_W = UtU * VtV  # Element-wise product
                
                # rhs_W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
                rhs_W = np.einsum('abc,ar,br->cr', tensor, U, V)
                
                reg = 1e-10 * np.eye(trial_rank)
                try:
                    W_new = np.linalg.solve(G_W + reg, rhs_W.T).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(G_W + reg, rhs_W.T, rcond=None)[0].T
                
                W = W_new
                
                # Update V similarly
                WtW = W.T @ W
                G_V = UtU * WtW
                
                rhs_V = np.einsum('abc,ar,cr->br', tensor, U, W)
                
                try:
                    V_new = np.linalg.solve(G_V + reg, rhs_V.T).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(G_V + reg, rhs_V.T, rcond=None)[0].T
                
                V = V_new
                
                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(recon - tensor)
                if error < 1e-12:
                    converged = True
                    break
            
            if not converged:
                continue
            
            # Try to round to rationals
            for factor in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
                V_r = np.round(V * factor) / factor
                W_r = np.round(W * factor) / factor
                U_r = np.round(U * factor) / factor
                
                if verify_decomposition(U_r, V_r, W_r, tensor, tol=1e-10):
                    return U_r, V_r, W_r, True
        
        return None, None, None, False
    
    # Try ranks from lower_bound to best_rank - 1
    for trial_rank in range(max(lower_bound, 10), best_rank):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(trial_rank, max_time=8.0)
        if result[3]:
            best_U, best_V, best_W, _ = result
            best_rank = trial_rank
            break
    
    # If we found a better rank, try to go even lower
    if best_rank < rank_trivial:
        for trial_rank in range(max(lower_bound, best_rank - 5), best_rank):
            if time.time() > deadline - 10:
                break
            
            result = try_rank(trial_rank, max_time=5.0)
            if result[3]:
                best_U, best_V, best_W, _ = result
                best_rank = trial_rank
                break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
