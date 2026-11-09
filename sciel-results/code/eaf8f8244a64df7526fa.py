"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_numpy(U_np, V_np, W_np, n, m, p, tol=1e-10):
    """Verify using numpy for speed."""
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k]) * W_np[k*n+i]
                # Actually need sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c]
                s = 0.0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_random_decomposition(n, m, p, target_rank, max_attempts=50, seed=42):
    """Try to find a random decomposition of given rank."""
    random.seed(seed)
    np.random.seed(seed)
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    for attempt in range(max_attempts):
        # Initialize with random sparse structure
        U[:] = 0
        V[:] = 0
        W[:] = 0
        
        for c in range(target_rank):
            # Randomly assign values with some sparsity
            for idx in range(n * m):
                if random.random() < 0.4:
                    U[idx, c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            for idx in range(m * p):
                if random.random() < 0.4:
                    V[idx, c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            for idx in range(p * n):
                if random.random() < 0.4:
                    W[idx, c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_structured_reduction(n, m, p, max_time=60.0):
    """Try structured approaches to reduce rank below n*m*p."""
    start_time = time.time()
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from a lower bound up to schoolbook
    # Theoretical lower bound is at least max(n*m, m*p, p*n)
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() - start_time > max_time:
            break
        
        # Try a few random attempts for this rank
        result = _try_random_decomposition(n, m, p, target_rank, max_attempts=20, 
                                           seed=random.randint(0, 10000))
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break  # Found the lowest rank, stop
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Check time
    if time.time() > deadline - 30:
        return best_result
    
    # Try to find a better decomposition
    result, rank = _try_structured_reduction(n, m, p, max_time=80.0)
    
    if result is not None and rank < best_rank:
        best_result = result
        best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
