import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact tensor decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    U_T = U_arr.T  # (rank, n*m)
    V_T = V_arr.T  # (rank, m*p)
    W_T = W_arr.T  # (rank, p*n)
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            s = np.sum(W_T[:, k * n + i] * U_T[:, a * m + b] * V_T[:, c * p + d])
                            if abs(s - expected) > tol:
                                return False
    return True

def _make_schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _try_random_sparse(n, m, p, R, seed):
    """Try a random sparse decomposition of rank R."""
    rng = random.Random(seed)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Each rank term should have some structure
    for r in range(R):
        # U_r: pick a few non-zero entries
        nnz_u = rng.randint(1, 3)
        for _ in range(nnz_u):
            idx = rng.randint(0, n * m - 1)
            U[idx, r] = rng.choice([-1, 1])
        
        # V_r: pick a few non-zero entries
        nnz_v = rng.randint(1, 3)
        for _ in range(nnz_v):
            idx = rng.randint(0, m * p - 1)
            V[idx, r] = rng.choice([-1, 1])
        
        # W_r: pick a few non-zero entries
        nnz_w = rng.randint(1, 3)
        for _ in range(nnz_w):
            idx = rng.randint(0, p * n - 1)
            W[idx, r] = rng.choice([-1, 1])
    
    return U, V, W

def _try_structured(n, m, p, R):
    """Try structured patterns for rank R."""
    results = []
    
    # Pattern: for n=2, try to pair up the two rows
    # Schoolbook has terms (i,j,k) for i in {0,1}, j in 0..4, k in 0..4
    # Try to merge (0,j,k) and (1,j',k') when they share structure
    
    # Attempt 1: Group by j, pair i=0 and i=1
    # For each j, we have 2*p = 10 schoolbook terms
    # Can we do it in fewer?
    
    # Attempt 2: Use the fact that for fixed j, the terms are:
    # (0,j,k) -> U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    # (1,j,k) -> U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
    # These share V[j*p+k]. Maybe we can combine?
    
    # For now, just return None and rely on random search
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from a lower bound up to best_rank - 1
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But actually the tensor rank lower bound is higher
    # For 2x5x5, known results suggest rank is around 25-30
    
    random.seed(12345)
    
    # Try each rank from 25 to best_rank-1
    for R in range(25, best_rank):
        if not still_going():
            break
        
        # Try multiple random attempts
        max_attempts = 50
        for attempt in range(max_attempts):
            if not still_going():
                break
            
            U, V, W = _try_random_sparse(n, m, p, R, seed=attempt * 1000 + R)
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = R
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break  # Found valid decomposition for this rank
        
        # If we found a valid rank, try to go lower
        if best_rank == R:
            continue
    
    return best
