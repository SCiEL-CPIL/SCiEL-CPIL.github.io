"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the full tensor M[a,b,c] with 1s at valid positions and 0s elsewhere
    # Valid positions: a=i*m+j, b=j*p+k, c=k*n+i for i in [0,n), j in [0,m), k in [0,p)
    
    # Total number of possible (a,b,c) tuples: n*m * m*p * p*n = 9*21*21 = 3969
    # But we only have 63 valid positions
    
    # The decomposition: M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # must equal 1 for valid (a,b,c) and 0 for invalid ones.
    
    # Strategy: Try to find a low-rank decomposition using alternating minimization.
    # We'll use a smooth objective: minimize sum of squared errors over ALL (a,b,c) tuples.
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        # Check all valid positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(U.shape[1]))
                    if abs(val - 1) > tol:
                        return False
        # Check some invalid positions (sample)
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            # Check if this is a valid position
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and j2 == k2:  # valid
                continue
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(U.shape[1]))
            if abs(val) > tol:
                return False
        return True
    
    def try_rank_als(r, max_iter=50, tol=1e-8):
        """Try to find a decomposition of rank r using ALS with smooth objective."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + r)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update W: for each c, solve least squares
            # For fixed U, V: minimize sum_{a,b} (M[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
            # For each c: sum_{a,b} (M[a,b,c] - W[c,:] @ (U[a,:] * V[b,:]))^2
            # This is: minimize ||M_c - W[c,:] @ A_c||^2 where A_c[a,b] = sum_r U[a,r] V[b,r]
            # But A_c depends on r, so this is not straightforward.
            
            # Alternative: use gradient descent or a different approach.
            # Let's use a simpler ALS: fix two matrices, solve for the third using linear least squares.
            
            # Fix U and V, solve for W:
            # For each c: minimize sum_{a,b} (M[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
            # Let B_c[r] = sum_{a,b} U[a,r] V[b,r] M[a,b,c]
            # Let C_c[r,s] = sum_{a,b} U[a,r] V[b,r] U[a,s] V[b,s]
            # Then solve C_c @ W[c,:] = B_c
            
            # But M[a,b,c] is 1 only for valid (a,b,c) and 0 otherwise.
            # This is complex. Let me try a different approach.
            
            break
        
        return None
    
    # For now, use the schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
