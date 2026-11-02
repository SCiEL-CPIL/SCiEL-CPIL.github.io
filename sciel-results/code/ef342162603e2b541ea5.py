"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[a, b, c] where:
    # a = i*m + j (index into A), b = j*p + k (index into B), c = k*n + i (index into C)
    # T[a, b, c] = 1 if there exists (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # We seek minimal rank r such that T[a,b,c] = sum_{r'} U[a,r']*V[b,r']*W[c,r']
    # Strategy: try ranks from lower bound upward, use iterative optimization
    
    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Objective: minimize ||T - sum_r U_r outer V_r outer W_r||^2
        # Flatten: objective = ||T.flatten() - sum_r (U[:,r] * V[:,r] * W[:,r]).flatten()||^2
        
        def objective(params):
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Compute tensor product sum
            # For efficiency, compute as: sum over r of outer(U[:,r], outer(V[:,r], W[:,r]))
            # This is expensive, so use matmul trick:
            # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # = (U * V) contracted with W... 
            # Efficient: for each r, add np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            # Or: T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            T_approx = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            return np.sum((T - T_approx) ** 2)
        
        # Start with schoolbook-like initialization for better convergence
        # Place non-zeros where the tensor is non-zero
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize: spread the 54 non-zero entries of T across rank columns
        # Each column r should capture some of the structure
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = col_idx % rank
                    U[i*m + j, r] += 1.0 / (54.0 / rank)  # scale
                    V[j*p + k, r] += 1.0 / (54.0 / rank)
                    W[k*n + i, r] += 1.0 / (54.0 / rank)
                    col_idx += 1
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B for optimization
        result = minimize(objective, params, method='L-BFGS-B', 
                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < 1e-6:
            U_final = result.x[:n*m*rank].reshape(n*m, rank)
            V_final = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_final = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer for cleanliness
            U_final = np.round(U_final * 2) / 2
            V_final = np.round(V_final * 2) / 2
            W_final = np.round(W_final * 2) / 2
            
            # Verify
            T_check = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
            if np.allclose(T_check, T, atol=1e-4):
                # Convert to list of lists
                return U_final.tolist(), V_final.tolist(), W_final.tolist()
            else:
                # Try without rounding
                T_check = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_check, T, atol=1e-6):
                    return U.tolist(), V.tolist(), W.tolist()
        return None
    
    # Try ranks from lower bound
    lower_bound = max(n*m, m*p, p*n)
    for rank in range(lower_bound, n*m*p + 1):
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
