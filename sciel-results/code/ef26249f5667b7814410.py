"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school), dtype=float)
    V_fb = np.zeros((m * p, rank_school), dtype=float)
    W_fb = np.zeros((p * n, rank_school), dtype=float)
    for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U_fb[i * m + j, idx] = 1
        V_fb[j * p + k, idx] = 1
        W_fb[k * n + i, idx] = 1
    
    def to_list(mat):
        # Convert to list of lists with clean integer/half-integer values
        result = []
        for row in mat:
            new_row = []
            for v in row:
                # Round to nearest half-integer
                r = round(v * 2) / 2
                if abs(r - round(r)) < 1e-9:
                    new_row.append(int(round(r)))
                else:
                    new_row.append(r)
            result.append(new_row)
        return result
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def solve_rank(r):
        """Try to find exact decomposition of rank r using random init + Newton."""
        if time.time() > deadline - 5:
            return None
        
        n, m, p = DIMENSIONS
        # Number of variables: r * (n*m + m*p + p*n)
        # Number of constraints: n * m * p
        # We need r >= n*m*p / (something) but realistically r needs to be substantial
        
        # Strategy: fix one matrix, solve for others
        # Let's try: random U, then solve V and W
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Use Newton's method to minimize ||F(U,V,W) - 1||^2
            # where F[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            # Simpler approach: use alternating least squares style
            # Fix W, solve for U and V; fix U,V, solve for W; iterate
            
            for iteration in range(50):
                if time.time() > deadline - 5:
                    return None
                
                # Fix W, solve for U and V
                # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # This is still nonlinear in U,V jointly
                
                # Alternative: use gradient descent on the loss
                # L = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Current value
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = val - 1
                            # Gradient contribution
                            for c in range(r):
                                dU[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
                
                # Learn rate schedule
                lr = 0.01 / (1 + iteration * 0.1)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if np.max(np.abs(dU)) < 1e-8 and np.max(np.abs(dV)) < 1e-8 and np.max(np.abs(dW)) < 1e-8:
                    break
            
            # Verify
            if verify(U, V, W, tol=1e-5):
                return (U, V, W)
        
        return None
    
    # Try ranks from known lower bounds upward
    # Known: rank >= max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Also Strassen-like bounds suggest we might do better than 84
    # Let's try a range of ranks
    
    best_result = None
    for r in [40, 45, 50, 55, 60, 65, 70, 75, 80]:
        if time.time() > deadline - 10:
            break
        
        result = solve_rank(r)
        if result is not None:
            best_result = (r, result)
            break  # Found a valid decomposition, stop
    
    if best_result is not None:
        r, (U, V, W) = best_result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback to schoolbook
    return {"U": to_list(U_fb), "V": to_list(V_fb), "W": to_list(W_fb)}
# EVOLVE-BLOCK-END
