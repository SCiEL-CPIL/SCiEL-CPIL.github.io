"""Matrix multiplication 4x5x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 125  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    best_rank = n * m * p  # schoolbook rank
    best_U, best_V, best_W = _schoolbook(n, m, p)

    # Try to find a lower-rank decomposition
    # Lower bound: max(n*p, m*p, n*m) = max(24, 30, 20) = 30
    # Try ranks from 30 upward
    
    for rank in range(30, best_rank + 1):
        if time.time() > deadline - 5:
            break
        result = _try_rank_als(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found the lowest rank we tried
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_rank_als(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition using alternating least squares."""
    # Initialize randomly
    np.random.seed(rank)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Build the list of valid (i,j,k) triples
    valid_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # ALS iterations
    for iteration in range(200):
        if time.time() > deadline - 10:
            return None
        
        # Update W: for each (i,k,c), W[k*n+i, c] should satisfy
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed (i,k), this is a linear system in W[k*n+i, :]
        # But it's coupled across c through the sum.
        # Actually for fixed (i,k): sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
        # This must hold for ALL j. So for fixed (i,k), we have m equations:
        # For each j: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
        # This is a system of m equations in rank unknowns W[k*n+i, 0..rank-1]
        # But we need it to be consistent, so the products U[i*m+j,c]*V[j*p+k,c] 
        # should be proportional across j for each c.
        
        # Simpler: just do one step of gradient descent on the loss
        # Loss = sum_{(i,j,k)} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # + lambda * sum_{invalid} (sum_c U[a,c]*V[b,c]*W[c_,c])^2
        
        # For efficiency, only use the "ones" constraints during optimization
        # and verify at the end.
        
        # Compute current residuals for valid triples
        residuals = np.zeros(len(valid_triples))
        for idx, (i, j, k) in enumerate(valid_triples):
            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            residuals[idx] = val - 1.0
        
        # Gradient descent step
        lr = 0.01
        for idx, (i, j, k) in enumerate(valid_triples):
            grad = residuals[idx]
            U[i*m+j] -= lr * grad * (V[j*p+k] * W[k*n+i])
            V[j*p+k] -= lr * grad * (U[i*m+j] * W[k*n+i])
            W[k*n+i] -= lr * grad * (U[i*m+j] * V[j*p+k])
        
        # Check convergence
        if np.max(np.abs(residuals)) < 1e-6:
            # Verify the full solution including zero constraints
            if _verify_decomposition(n, m, p, U, V, W):
                # Convert to list format
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition satisfies all tensor constraints."""
    rank = U.shape[1]
    
    # Check ones constraints
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-4:
                    return False
    
    # Check zero constraints (sample a subset for speed)
    # For (a,b,c) where indices don't align
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c_idx = random.randint(0, p*n-1)
        
        # Decode indices
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c_idx // n
        i_c = c_idx % n
        
        # Check if this is a valid triple
        if i_a == i_c and j_a == j_b and j_b == k_b:
            continue  # This is a valid triple, skip
        
        val = np.sum(U[a] * V[b] * W[c_idx])
        if abs(val) > 1e-4:
            return False
    
    return True

# EVOLVE-BLOCK-END
