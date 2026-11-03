"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    # True tensor: T[i, j, k] = 1 for all i,j,k (matrix mult tensor)
    # Shape: (n, m, p) with T[i,j,k] = 1
    
    # Try to find a low-rank decomposition
    # Start with a reasonable upper bound and try to reduce
    
    # First, get the schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    def build_schoolbook(rank):
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
        return {"U": U, "V": V, "W": W}
    
    # Try various ranks from low to high
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(24, 32, 12) = 32
    # But actual rank could be higher
    
    best_result = build_schoolbook(rank_schoolbook)
    best_rank = rank_schoolbook
    
    # Try to find decompositions with lower rank
    # Use randomized initialization and check if we can satisfy the tensor equation
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize U, V, W with small random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # The tensor equation: for all i,j,k:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want this to equal 1 for all i,j,k
        
        # Use alternating least squares style approach
        # For fixed V, W: solve for U
        # For fixed U, W: solve for V
        # For fixed U, V: solve for W
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a system of p equations in r unknowns
            # Let A[i,j,k] = sum_r V[j*p+k, r] * W[k*n+i, r]
            # Then U[i*m+j, :] should satisfy: A @ U[i*m+j, :]^T = 1 (vector of ones)
            # Actually: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            # So: (V[j*p+k, :] * W[k*n+i, :]) @ U[i*m+j, :]^T = 1 for each k
            
            for i in range(n):
                for j in range(m):
                    # Build matrix: rows k, cols r: M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M @ U[i*m+j, :]^T = ones(p)
                    # Use least squares
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 100 == 0:
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0) ** 2
                if err < 1e-10:
                    break
        
        # Final check
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        
        if err < 1e-8:
            # Convert to list of lists with rounded values
            U_list = [[round(float(U[i, r]), 6) for r in range(r)] for i in range(n*m)]
            V_list = [[round(float(V[i, r]), 6) for r in range(r)] for i in range(m*p)]
            W_list = [[round(float(W[i, r]), 6) for r in range(r)] for i in range(p*n)]
            
            # Verify with the rounded values
            verify_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_list[i*m+j][rr] * V_list[j*p+k][rr] * W_list[k*n+i][rr] for rr in range(r))
                        verify_err += (val - 1.0) ** 2
            
            if verify_err < 1e-6:
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound is max(n*p, m*p, n*m) = max(24, 32, 12) = 32
    # But let's try a range
    min_rank_to_try = 32
    max_rank_to_try = min(60, rank_schoolbook)
    
    for r in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline:
                break
            
            result = try_rank(r, max_iters=3000, seed=seed)
            if result is not None:
                best_rank = r
                best_result = result
                break
        
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END

