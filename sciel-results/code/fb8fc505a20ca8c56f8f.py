import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # The tensor T is defined by T[i, j, k] = 1 if the indices are valid for the multiplication.
    # For 3x3 * 3x5, the output is 3x5.
    # The tensor has dimensions (n, m, p) = (3, 3, 5).
    # The decomposition is T[i, j, k] = sum_r U[i, m+j, r] * V[j, p+k, r] * W[k, n+i, r]
    # Wait, the indices in the problem statement are:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    # And the tensor entry is T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's verify the tensor structure.
    # Matrix multiplication C = A * B, where A is n x m, B is m x p.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has entries T[i, j, k] which is 1 if we are using A[i, j] and B[j, k] to compute C[i, k].
    # So T[i, j, k] = 1 for all i, j, k.
    
    # We want to find U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # We will try ranks from a low value upwards.
    # Known lower bound for 3x3x5 is 24? Let's check.
    # Rank of 3x3x3 is 23 (Strassen).
    # Rank of 3x3x4 is 24?
    # Rank of 3x3x5 is likely 24 or 25.
    
    # We'll start with rank 24 and go up to 45 (schoolbook).
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute the tensor indices
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_indices.append((i, j, k))
                
    def check_solution(U, V, W, R):
        for i, j, k in tensor_indices:
            s = 0
            for r in range(R):
                s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            if s != 1:
                return False
        return True
        
    def random_init(R):
        U = np.zeros((n*m, R), dtype=int)
        V = np.zeros((m*p, R), dtype=int)
        W = np.zeros((p*n, R), dtype=int)
        for r in range(R):
            # Initialize with random -1, 0, 1
            for idx in range(n*m):
                U[idx, r] = random.choice([-1, 0, 1])
            for idx in range(m*p):
                V[idx, r] = random.choice([-1, 0, 1])
            for idx in range(p*n):
                W[idx, r] = random.choice([-1, 0, 1])
        return U, V, W
        
    def local_search(U, V, W, R, max_steps=1000):
        # Try to fix the solution by flipping random entries
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            # Check if already solved
            if check_solution(U, V, W, R):
                return U, V, W
                
            # Pick a random tensor entry that is not satisfied
            errors = []
            for i, j, k in tensor_indices:
                s = 0
                for r in range(R):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if s != 1:
                    errors.append((i, j, k))
                    
            if not errors:
                return U, V, W
                
            i, j, k = random.choice(errors)
            # Pick a random rank r and a random matrix to update
            r = random.randint(0, R-1)
            mat = random.choice([0, 1, 2]) # 0: U, 1: V, 2: W
            if mat == 0:
                idx = i*m + j
                U[idx, r] = random.choice([-1, 0, 1])
            elif mat == 1:
                idx = j*p + k
                V[idx, r] = random.choice([-1, 0, 1])
            else:
                idx = k*n + i
                W[idx, r] = random.choice([-1, 0, 1])
                
        return None
        
    # Try ranks from 24 to 45
    for R in range(24, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for each rank
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            U, V, W = random_init(R)
            result = local_search(U, V, W, R, max_steps=500)
            if result is not None:
                U, V, W = result
                # Convert to lists of lists as required
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        if best_rank == R:
            break # Found a solution for this rank, which is the minimum we tried
            
    # If no solution found, fall back to schoolbook
    if best_U is None:
        R = best_rank
        U = np.zeros((n*m, R), dtype=int)
        V = np.zeros((m*p, R), dtype=int)
        W = np.zeros((p*n, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        best_U = U.tolist()
        best_V = V.tolist()
        best_W = W.tolist()
        
    return {"U": best_U, "V": best_V, "W": best_W}
