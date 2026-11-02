"""Matrix multiplication 2x3x9 rank-minimal decomposition using Kronecker structure."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    We attempt to find a decomposition with rank less than 54 using a structured
    approach. The minimum possible rank is at least max(n*m, m*p, p*n) = 27.
    We try rank 27 first, then increase if needed.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 120
    
    # Try to find a low-rank decomposition
    # Start with rank 27 (the theoretical lower bound)
    for rank in range(27, 55):
        if time.time() > deadline - 5:
            break
        
        # Initialize with schoolbook pattern for the first 'rank' terms
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Try to find a valid decomposition using the tensor equation:
        # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
        # where delta is 1 if this is a valid multiplication term, 0 otherwise
        
        # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
        # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor
        # in the appropriate subspace)
        
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 if the indices
        # correspond to a valid multiplication term. For (n,m,p) = (2,3,9):
        # T[i,j,k] = 1 for all i in [0,n), j in [m), k in [0,p)
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # This is a system of equations. Let's try to solve it.
        
        # Number of equations: n*m*p = 54
        # Number of unknowns: rank * (n*m + m*p + p*n) = rank * (6 + 27 + 18) = rank * 51
        
        # For rank >= 54/51 > 1, we have more unknowns than equations, so a solution
        # should exist. Let's try rank = 27.
        
        # Use a simple initialization and Newton's method
        np.random.seed(42)
        U_init = np.random.randn(n * m, rank) * 0.1
        V_init = np.random.randn(m * p, rank) * 0.1
        W_init = np.random.randn(p * n, rank) * 0.1
        
        # Define the residual: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        def residual(U, V, W):
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
            return res
        
        def residual_flat(U, V, W):
            return residual(U, V, W).flatten()
        
        # Try a few random initializations
        found = False
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Use gradient descent
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                res = residual(U, V, W)
                if np.max(np.abs(res)) < 1e-10:
                    found = True
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = res[i,j,k]
                            dU[i*m+j, :] += r * V[j*p+k, :] * W[k*n+i, :]
                            dV[j*p+k, :] += r * U[i*m+j, :] * W[k*n+i, :]
                            dW[k*n+i, :] += r * U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if found:
                break
        
        if found:
            # Verify the solution
            res = residual(U, V, W)
            if np.max(np.abs(res)) < 1e-8:
                # Round to nearest half-integer if close
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded version still works
                res_rounded = residual(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(res_rounded)) < 1e-6:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                else:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Fallback: schoolbook decomposition with rank 54
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
