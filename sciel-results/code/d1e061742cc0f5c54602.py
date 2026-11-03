"""
Tensor decomposition for 3x3x8 matrix multiplication.

Approach:
- The current program returns the schoolbook decomposition of rank 72.
- This new program attempts to find a lower rank decomposition using stochastic gradient descent.
- It defines the matrix multiplication tensor T (which is all ones for the standard multiplication)
  and tries to find factors U, V, W of rank r < 72 such that their tensor product sum equals T.
- If a valid low-rank decomposition is found, it is returned.
- Otherwise, it falls back to the schoolbook rank 72.
"""

import numpy as np
import random

DIMENSIONS = [3, 3, 8]

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(rank, max_iters=5000, lr=0.01, seed=42):
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(seed)
    
    # Initialize with small random values
    U = rng.normal(0, 0.1, (n * m, rank))
    V = rng.normal(0, 0.1, (m * p, rank))
    W = rng.normal(0, 0.1, (p * n, rank))
    
    # Target tensor is all ones
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Flatten indices for vectorized operations if possible, or use loops for clarity given small sizes
    # n=3, m=3, p=8 -> 72 terms.
    
    for it in range(max_iters):
        # Compute current approximation
        # Approx[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # To compute gradients efficiently, we can compute the error tensor E
        # E[i,j,k] = Approx[i,j,k] - 1
        
        # Gradient for U[i,j,c]:
        # dL/dU[i,j,c] = sum_{j',k'} E[i,j',k'] * V[j',k',c] * W[k',i,c]
        # Wait, the indices in the sum for U[i,j,c] are over the other indices j' and k' such that the term is used.
        # The term involving U[i,j,c] is only in the sum for the specific (i,j,k) where U has index (i,j).
        # Actually, U[i*m+j, c] appears in the term for (i, j, k) for ALL k.
        # So, for a fixed i, j, c:
        # U[i*m+j, c] is multiplied by V[j*p+k, c] * W[k*n+i, c] for each k.
        # The error E[i,j,k] corresponds to the sum over c.
        # So, dL/dU[i*m+j, c] = sum_{k} E[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        
        # Similarly:
        # dL/dV[j*p+k, c] = sum_{i} E[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
        # dL/dW[k*n+i, c] = sum_{j} E[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
        
        # Let's compute E first
        E = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    E[i, j, k] = s - 1.0
                    
        # Compute Gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = E[i, j, k]
                    for c in range(rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        w_val = W[k*n+i, c]
                        
                        # Contribution to grad_U[i*m+j, c]
                        grad_U[i*m+j, c] += e * v_val * w_val
                        # Contribution to grad_V[j*p+k, c]
                        grad_V[j*p+k, c] += e * u_val * w_val
                        # Contribution to grad_W[k*n+i, c]
                        grad_W[k*n+i, c] += e * u_val * v_val
                        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if np.max(np.abs(E)) < 1e-6:
            return {"U": U, "V": V, "W": W}
            
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a lower rank decomposition
    # Start from a reasonable lower bound. 
    # For n=3, m=3, p=8, schoolbook is 72.
    # We try ranks from 60 down to 50. If we find one, great.
    # If not, fall back to schoolbook.
    
    # Note: Gradient descent might get stuck in local minima.
    # We try a few seeds and ranks.
    
    for r in range(60, 49, -1):
        for seed in range(10):
            res = _try_find_decomposition(r, max_iters=2000, lr=0.05, seed=seed)
            if res is not None:
                # Convert to list of lists if required, but numpy arrays are usually accepted or convertible
                # The task says "factor matrices... with integer/half-integer entries".
                # Gradient descent produces floats. We might need to round?
                # "exactly decompose" implies the values should be exact.
                # If the error is < 1e-6, rounding to nearest integer or half-integer might work if the true solution is rational.
                # However, tensor decompositions of matrix multiplication often have integer or rational coefficients.
                # Let's check if we can round the values to nearest 0.5 or 1.
                
                U_rounded = np.round(res["U"] * 2) / 2
                V_rounded = np.round(res["V"] * 2) / 2
                W_rounded = np.round(res["W"] * 2) / 2
                
                # Verify the rounded solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                            if abs(s - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
                
                if valid:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    
    # Fallback to schoolbook
    res = _schoolbook_decomposition()
    return {
        "U": res["U"].tolist(),
        "V": res["V"].tolist(),
        "W": res["W"].tolist()
    }
# EVOLVE-BLOCK-END
