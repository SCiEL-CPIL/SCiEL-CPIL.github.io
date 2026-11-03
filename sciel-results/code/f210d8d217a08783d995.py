"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index conventions:
    #   U row index: i*m + j  (i in 0..n-1, j in 0..m-1)
    #   V row index: j*p + k  (j in 0..m-1, k in 0..p-1)
    #   W row index: k*n + i  (k in 0..p-1, i in 0..n-1)
    
    # Schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Function to verify a decomposition
    def verify(U, V, W, rank):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Function to compute error tensor efficiently
    def compute_error(U, V, W, rank):
        """Compute error[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        error = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error[i,j,k] = 1.0 - val
        return error
    
    # Function to check if all errors are below tolerance
    def is_valid(error, tol=1e-9):
        return np.all(np.abs(error) < tol)
    
    # Schoolbook decomposition
    def schoolbook_decomp():
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
        return U, V, W, rank
    
    # Try structured low-rank constructions first
    
    # Strategy 1: Exploit p=6 as two 3x3 blocks
    # C = A * B where B = [B1 | B2] with B1, B2 being 3x3
    # C1 = A*B1, C2 = A*B2
    # If we have a rank-r decomposition for 3x3*3x3, we can get rank-2r for 3x3*3x6
    # But we might do better by sharing some computations
    
    # Strategy 2: Try to find a direct low-rank decomposition
    
    # Candidate ranks to try (from low to high)
    candidate_ranks = [18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 44, 46, 48, 50, 52]
    
    # For each target rank, try multiple random initializations
    max_init_per_rank = 5
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        for init_idx in range(max_init_per_rank):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            np.random.seed(42 + target_rank * 100 + init_idx)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Iterative refinement using gradient descent
            eta = 0.05
            for iteration in range(200):
                if time.time() > deadline:
                    break
                    
                # Compute error
                error = compute_error(U, V, W, target_rank)
                
                # Check if valid
                if is_valid(error, 1e-9):
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
                
                # Compute gradients
                # dU[i*m+j, r] = -sum over (k) error[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                # dV[j*p+k, r] = -sum over (i) error[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                # dW[k*n+i, r] = -sum over (j) error[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = error[i,j,k]
                            if abs(err) < 1e-12:
                                continue
                            for r in range(target_rank):
                                dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                
                # Update
                U -= eta * dU
                V -= eta * dV
                W -= eta * dW
                
                # Adaptive learning rate
                if iteration % 20 == 0:
                    eta *= 0.95
            
            if best_rank <= target_rank:
                break
    
    # If no better decomposition found, use schoolbook
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
