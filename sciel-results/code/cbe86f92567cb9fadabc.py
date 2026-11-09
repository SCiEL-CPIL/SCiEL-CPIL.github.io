"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x5 matrix.
    
    The tensor T[i, j, k] for matrix multiplication C = A * B where A is n x m
    and B is m x p satisfies:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The multiplication tensor has entries T[i, j, k] = 1 if the product A[i,j]*B[j,k]
    contributes to C[i,k], which it always does. So T[i, j, k] = 1 for all valid (i, j, k).
    
    Wait, let me reconsider. The matrix multiplication tensor T has T[i, j, k] = 1 
    for all i in [0,n), j in [0,m), k in [0,p). The rank is the minimum number of 
    rank-1 tensors that sum to T.
    
    A rank-1 tensor is U[:, c] * V[:, c] * W[:, c] (outer product).
    
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k] = 1
    
    This is a non-linear system. Let's use the approach of trying different ranks 
    and checking feasibility.
    
    For a given rank r, we want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a system of n*m*p = 60 equations in 3*r variables (but bilinear/quadratic).
    
    Let's try a different approach. We can fix some variables and solve for others.
    
    Actually, let's use the known lower bound. For matrix multiplication of n x m by m x p,
    the rank is at least max(n*p, m*p, n*m) = max(15, 20, 12) = 20.
    
    The schoolbook algorithm has rank n*m*p = 60.
    
    Let's try to find a decomposition with rank < 60 using optimization.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    # The tensor T[i, j, k] = 1 for all i, j, k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Try ranks from a lower bound upward, and for each rank,
    # use a non-linear solver or iterative method to find a valid decomposition.
    
    # Lower bound: max(n*p, m*p, n*m) = 20
    # Let's start trying from rank 20 upward.
    
    # For a fixed rank r, we have variables U (12 x r), V (20 x r), W (15 x r)
    # Total variables: 12r + 20r + 15r = 47r
    
    # Constraints: for each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Number of constraints: n*m*p = 60
    
    # This is a system of 60 equations in 47r unknowns (with product terms).
    
    # Let's use scipy.optimize.least_squares or a custom approach.
    
    # Alternative: We can use the fact that if we fix W, then the problem becomes
    # bilinear in U and V. Let's try an iterative approach.
    
    # Actually, let's try a simpler approach first: use the known Strassen-like 
    # decompositions or just try to optimize.
    
    # Let's use a non-linear least squares approach to minimize the residual
    # sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    # We'll try different ranks and see which one achieves zero residual.
    
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    # Try ranks from 20 upward
    for rank in range(20, 61):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values
        np.random.seed(rank)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Use scipy's least_squares to minimize the residual
        def residual(params):
            U_flat = params[:nm*rank].reshape(nm, rank)
            V_flat = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_flat = params[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all i,j,k
            # This is a 3D tensor contraction
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Efficient computation using einsum
            # U has shape (nm, rank) where nm = n*m, index i*m+j
            # V has shape (mp, rank) where mp = m*p, index j*p+k
            # W has shape (pn, rank) where pn = p*n, index k*n+i
            
            # We want T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Let's compute this step by step
            # For each c, we have a rank-1 tensor U_c * V_c * W_c
            # T = sum_c outer(U_c, V_c, W_c) but with specific index mappings
            
            # U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
            
            # Let's use einsum
            # First, reshape U to (n, m, rank)
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            
            # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
            # This is a contraction over j and c
            # T[i,k] = sum_j sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
            
            # But we need T[i,j,k] = 1 for all i,j,k, not just T[i,k]
            
            # Wait, I think I'm confusing the tensor definition.
            
            # The matrix multiplication tensor T has T[i, j, k] = 1 for all valid indices.
            # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # So for each (i, j, k), we need the sum to equal 1.
            
            # Let's compute this efficiently
            # For fixed i, k: sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # But we need it for each (i, j, k) separately, not summed over j.
            
            # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Let's compute this as a 3D array
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Using einsum:
            # U has indices (i*m+j, c) -> let's call the first index 'a' where a = i*m+j
            # V has indices (j*p+k, c) -> let's call the first index 'b' where b = j*p+k  
            # W has indices (k*n+i, c) -> let's call the first index 'd' where d = k*n+i
            
            # We need to sum over c: U[a, c] * V[b, c] * W[d, c]
            # But a, b, d are related to i, j, k.
            
            # Let's just compute it directly for small sizes
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        d = k * n + i
                        T[i, j, k] = np.sum(U_flat[a, :] * V_flat[b, :] * W_flat[d, :])
            
            # Residual is T - 1
            return (T - 1.0).flatten()
        
        # Initial parameters
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try to solve
        try:
            result = scipy_least_squares(residual, x0, method='lm', max_nfev=1000)
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                # Check if we have a valid solution
                final_residual = np.max(np.abs(result.fun))
                if final_residual < 1e-8:
                    U_sol = result.x[:nm*rank].reshape(nm, rank)
                    V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Convert to list format
                    return {
                        "U": U_sol.tolist(),
                        "V": V_sol.tolist(),
                        "W": W_sol.tolist()
                    }
        except Exception:
            pass
        
        # Try with different initializations
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            np.random.seed(seed * 1000 + rank)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = scipy_least_squares(residual, x0, method='lm', max_nfev=500)
                if result.success or np.max(np.abs(result.fun)) < 1e-8:
                    final_residual = np.max(np.abs(result.fun))
                    if final_residual < 1e-8:
                        U_sol = result.x[:nm*rank].reshape(nm, rank)
                        V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                        W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                pass
    
    # If we couldn't find a better decomposition, return the schoolbook one
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# Import scipy at module level
from scipy.optimize import least_squares as scipy_least_squares
# EVOLVE-BLOCK-END
