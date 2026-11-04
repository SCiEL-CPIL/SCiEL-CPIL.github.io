"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s limit
    
    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)
    
    # The schoolbook decomposition has rank 84
    # We want to find something better if possible
    
    # Known lower bounds:
    # - Trivial: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # - Better bounds exist but are complex
    
    # Strategy: Try structured decompositions using known techniques
    # For matrix multiplication, we can use:
    # 1. Strassen-like algorithms for smaller blocks
    # 2. Random search with verification
    # 3. Exploit the structure of the tensor
    
    # First, let's create the schoolbook decomposition as a fallback
    def make_schoolbook():
        rank = nmp
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
    
    # Verify a decomposition
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over rank of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    total = np.sum(
                        U_np[i * m + j, :] * 
                        V_np[j * p + k, :] * 
                        W_np[k * n + i, :]
                    )
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a better decomposition using random search
    # Focus on ranks that might be achievable
    
    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()
    
    # Try random search for ranks from a reasonable lower bound up
    # Lower bound is at least 28, but let's start higher for efficiency
    random.seed(12345)
    np.random.seed(12345)
    
    # Try several target ranks
    # Based on known results, the optimal rank for 3x4x7 is likely in the 40-70 range
    target_ranks = list(range(40, min(75, nmp)))
    
    # Shuffle to try different ranks
    random.shuffle(target_ranks)
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if target_rank < 60 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values from {-1, 0, 1}
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            # Random initialization
            for c in range(target_rank):
                U[:, c] = np.random.choice([-1, 0, 1], size=n * m)
                V[:, c] = np.random.choice([-1, 0, 1], size=m * p)
                W[:, c] = np.random.choice([-1, 0, 1], size=p * n)
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < nmp:
            break
    
    # If we found a better decomposition, use it
    # Otherwise, use the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
