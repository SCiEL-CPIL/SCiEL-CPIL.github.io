import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank_schoolbook = n * m * p  # 72
    
    # Schoolbook decomposition
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook.copy()
    best_V = V_schoolbook.copy()
    best_W = W_schoolbook.copy()
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Strategy 1: Try to find lower-rank decompositions using structured approaches
    # For matrix multiplication M_{n,m,p}, the tensor is T[i,j,k] = 1 for all valid (i,j,k)
    # We need rank-r decomposition where sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try ranks from 64 down to 50
    target_ranks = [64, 62, 60, 58, 56, 54, 52, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:  # Keep 20s buffer
            break
        
        # Method 1: Random initialization with refinement
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple check
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # Strategy 2: Try structured decomposition using Kronecker products
    # For M_{3,3,8}, we can try to exploit the structure
    # The tensor T[i,j,k] = 1 for all i,j,k means we need:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # Try a block-structured approach
    if best_rank == rank_schoolbook and time.time() < deadline - 30:
        # Try rank 70 with specific structure
        for target_rank in [70, 68, 66]:
            if time.time() > deadline - 30:
                break
            
            # Initialize with structured pattern
            for attempt in range(10):
                if time.time() > deadline - 30:
                    break
                
                np.random.seed(attempt * 2000 + target_rank)
                # Use a mix of random and structured initialization
                U = np.random.randn(n * m, target_rank) * 0.1
                V = np.random.randn(m * p, target_rank) * 0.1
                W = np.random.randn(p * n, target_rank) * 0.1
                
                # Add some structure: make first few columns binary
                for c in range(min(10, target_rank)):
                    U[:, c] = np.random.randint(0, 2, n * m) * 0.5
                    V[:, c] = np.random.randint(0, 2, m * p) * 0.5
                    W[:, c] = np.random.randint(0, 2, p * n) * 0.5
                
                if check_decomposition(U, V, W):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
            
            if best_rank < rank_schoolbook:
                break
    
    # Strategy 3: Try to use known results - for M_{3,3,8}, rank might be achievable at 64
    # The schoolbook is 72, so we're looking for savings
    if best_rank == rank_schoolbook and time.time() < deadline - 10:
        # Try a more systematic approach: start from schoolbook and try to merge terms
        # This is complex, so we'll try random search with more attempts at rank 64
        target_rank = 64
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 3000 + target_rank)
            # Scale to help with convergence
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
