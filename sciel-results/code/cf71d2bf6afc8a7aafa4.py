"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 3x5x6 matrix multiplication tensor."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x5x6."""
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS  # 3, 5, p = 6
    schoolbook_U, schoolbook_V, schoolbook_W, schoolbook_rank = _schoolbook(n, m, p)
    
    best_U = schoolbook_U
    best_V = schoolbook_V
    best_W = schoolbook_W
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition using random search + local optimization
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of 90 equations. We can parameterize and use least squares.
    
    # Strategy: try ranks from 89 down to some lower bound
    # For each rank, try random initializations and solve the nonlinear system
    
    np.random.seed(42)
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actually rank >= n*m*p / max_possible_product_per_term... 
    # A better lower bound: rank >= ceil(n*m*p / (max|U|*max|V|*max|W|)) but with integers this is tricky
    # Known: rank(T_{n,m,p}) >= n*m + n*p + m*p - 2 for some cases, but not always
    # For now, try ranks from 89 down to 50
    
    for target_rank in range(schoolbook_rank - 1, 49, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U_rand = np.random.randn(n * m, target_rank) * 0.5
            V_rand = np.random.randn(m * p, target_rank) * 0.5
            W_rand = np.random.randn(p * n, target_rank) * 0.5
            
            # Use iterative refinement: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is nonlinear. Use a simple iterative approach:
            # Fix two matrices, solve for the third via linear system
            
            # Alternate: fix V and W, solve for U; fix U and W, solve for V; fix U and V, solve for W
            # Each solve is a linear system
            
            U_curr = U_rand.copy()
            V_curr = V_rand.copy()
            W_curr = W_rand.copy()
            
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c]*W[k*n+i,c] should contribute to T[i,j,k]
                # Actually, for fixed (i,j), the equation is: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
                # This is: U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) = 1 for each k
                # So U[i*m+j,:] @ M_{i,j,k} = 1 where M_{i,j,k}[c] = V[j*p+k,c]*W[k*n+i,c]
                # We have p equations for n*m unknowns per (i,j) row... wait no.
                
                # For each (i,j), we have p equations (one for each k) and target_rank unknowns (U[i*m+j,0..rank-1])
                # So we have an underdetermined system. Use least squares.
                
                # Build the system: for each (i,j), A_{i,j} is p x rank, b_{i,j} is p x 1 of ones
                A_U = np.zeros((n*m*p, target_rank))
                b_U = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A_U[idx, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                            idx += 1
                
                # Solve least squares: A_U @ u = b_U where u is flattened U
                U_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
                U_curr = U_flat.reshape(n*m, target_rank)
                
                # Fix U and W, solve for V
                A_V = np.zeros((n*m*p, target_rank))
                b_V = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A_V[idx, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                            idx += 1
                
                V_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
                V_curr = V_flat.reshape(m*p, target_rank)
                
                # Fix U and V, solve for W
                A_W = np.zeros((n*m*p, target_rank))
                b_W = np.ones(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A_W[idx, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                            idx += 1
                
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                W_curr = W_flat.reshape(p*n, target_rank)
                
                # Check convergence
                # Compute the residual: for each (i,j,k), |sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1|
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Verify with exact arithmetic
                U_list = U_curr.tolist()
                V_list = V_curr.tolist()
                W_list = W_curr.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    best_rank = target_rank
                    break  # Found a valid decomposition at this rank, try lower
                else:
                    # Numerical solution but not exact - try to round to nearest integers/half-integers
                    U_rounded = np.round(U_curr * 2) / 2
                    V_rounded = np.round(V_curr * 2) / 2
                    W_rounded = np.round(W_curr * 2) / 2
                    
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        best_rank = target_rank
                        break
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Verify final result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U = schoolbook_U
        best_V = schoolbook_V
        best_W = schoolbook_W
        best_rank = schoolbook_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
