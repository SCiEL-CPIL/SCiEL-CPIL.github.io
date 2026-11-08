"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # T_{i,j,k} for i in [0,n), j in [0,m), k in [0,p)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    
    # We need: for all i,j,k: delta(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where delta(i,j,k) = 1 if this is a valid "edge" in the tensor, else 0
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # and the decomposition must satisfy: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # The tensor is: T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,9)
    # We need rank R such that we can write T as sum of R rank-1 tensors
    
    # Use known result: for 4x4 x 4x4, rank is at most 21 (Winograd)
    # For 4x4 x 4x9, we can split the 9 into 4+4+1
    # Let's try to construct using block decomposition
    
    # Strategy: Use the fact that we can decompose the multiplication
    # A (4x4) * B (4x9) = A * [B1 | B2 | B3] where B1,B2 are 4x4, B3 is 4x1
    
    # For the 4x4 x 4x4 part, use a rank-21 decomposition
    # For the 4x4 x 4x1 part, use a rank-4 decomposition (schoolbook for 4x4*4x1)
    
    # But we need to combine them. The left factor A is shared.
    
    # Actually, let me use a numerical approach with a reasonable rank.
    # Known lower bound for 4x4x4 is 21, so for 4x4x9 it should be less than 144.
    
    # Let's try rank = 60 as a starting point and optimize
    
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Try different ranks, starting from a reasonable estimate
    # For 4x4 x 4x9, a good upper bound might be around 50-70
    
    best_rank = n * m * p  # schoolbook as fallback
    best_decomp = None
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, R
    
    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = R_sb
    
    # Try to find better decomposition using optimization
    # The constraint is: for all i,j,k in valid range:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of equations. Let's use iterative optimization.
    
    # Try ranks from a lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) / something, but more tightly:
    # The tensor has n*m*p = 144 non-zero entries, so rank >= 144 / max_entry_product... 
    # Actually, rank >= n*m*p / (something). Let's just try from 40 upward.
    
    for R in range(40, min(100, best_rank)):
        if time.time() - start_time > deadline - 30:
            break
        
        # Initialize with random values
        np.random.seed(42 + R)
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # Use iterative optimization: minimize the error
        # Error = sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1.0) ** 2
            return err
        
        # Simple gradient descent
        lr = 0.01
        for it in range(2000):
            if time.time() - start_time > deadline - 30:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = s - 1.0
                        for r in range(R):
                            dU[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 200 == 0:
                e = compute_error(U, V, W)
                if e < 1e-10:
                    break
        
        e = compute_error(U, V, W)
        if e < 1e-10:
            best_rank = R
            best_decomp = (U, V, W)
            break
    
    # Convert to lists
    U, V, W = best_decomp
    U_list = [[float(U[i, r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

