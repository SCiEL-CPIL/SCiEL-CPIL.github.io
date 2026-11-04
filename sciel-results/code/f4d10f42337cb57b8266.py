import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # if this is a valid multiplication entry, 0 otherwise
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This is always a valid entry in matrix multiplication
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T[i,j,k] = 1 for all valid (i,j,k)
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, time_limit=10.0, seed=42):
    """Try to find a decomposition of given rank."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    best_solution = None
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random half-integers
        U = rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n*m, rank))
        V = rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m*p, rank))
        W = rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p*n, rank))
        
        # Build the tensor equation system
        # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of n*m*p equations
        
        # Use iterative refinement: randomly pick equations and try to satisfy them
        success = True
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            target = 1.0
            
            current = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            if abs(current - target) < 1e-9:
                continue
            
            # Try to adjust one of the three factors
            # Pick a random rank index to modify
            r = rng.randint(rank)
            
            # Compute gradient-like update
            # d(U[i*m+j, r]) = V[j*p+k, r] * W[k*n+i, r]
            # d(V[j*p+k, r]) = U[i*m+j, r] * W[k*n+i, r]
            # d(W[k*n+i, r]) = U[i*m+j, r] * V[j*p+k, r]
            
            gU = V[j*p+k, r] * W[k*n+i, r]
            gV = U[i*m+j, r] * W[k*n+i, r]
            gW = U[i*m+j, r] * V[j*p+k, r]
            
            # Pick the factor with largest gradient
            grads = [abs(gU), abs(gV), abs(gW)]
            if max(grads) < 1e-12:
                continue
            
            # Try small adjustments
            for delta in [0.5, -0.5, 1.0, -1.0]:
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                
                if abs(gU) == max(grads):
                    new_U[i*m+j, r] += delta
                elif abs(gV) == max(grads):
                    new_V[j*p+k, r] += delta
                else:
                    new_W[k*n+i, r] += delta
                
                # Check if this improves things
                new_current = np.sum(new_U[i*m+j, :] * new_V[j*p+k, :] * new_W[k*n+i, :])
                if abs(new_current - target) < abs(current - target):
                    U, V, W = new_U, new_V, new_W
                    break
        
        # Final verification
        if _verify(U, V, W, n, m, p, rank):
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try ranks from low to high
    # Known lower bound for 4x5x6 is at least around 20-30
    # Try a range of ranks
    for rank in range(20, 121):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(rank, n, m, p, time_limit=12.0, seed=rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}
