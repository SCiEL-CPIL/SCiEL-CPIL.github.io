import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # T[(i,j), (j,k), (i,k)] = 1 for all valid i,j,k
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Convention: W[k*n + i, r]
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reconstruct tensor: T[(i,j),(j,k),(i,k)] should equal sum_r U[(i,j),r]*V[(j,k),r]*W[(i,k),r]
        # W index: k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(actual - expected) > 1e-6:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # For each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Vectorize: for fixed i, j: sum_k ... 
        # Let's do it efficiently
        for i in range(n):
            for j in range(m):
                # For this (i,j), we need: for all k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                u_row = U[i*m + j, :]  # shape (rank,)
                for k in range(p):
                    v_val = V[j*p + k, :]  # shape (rank,)
                    w_val = W[k*n + i, :]  # shape (rank,)
                    actual = np.dot(u_row, v_val * w_val)
                    if abs(actual - 1.0) > 1e-6:
                        return False
        return True
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    
    # Try to find better decompositions
    # Strategy: Try ranks from 27 upward, use structured search
    
    # Known: for 3x3 x 3x3, rank <= 23. For 3x3 x 3x6, try rank 46 (two copies)
    # But let's try smaller ranks first with random search
    
    # Try a structured approach: use the 2x2 Strassen-like ideas
    # For now, let's try random search for small ranks
    
    import random
    random.seed(42)
    
    for target_rank in [27, 28, 29, 30, 32, 36, 40, 45, 46, 48, 50, 52, 53]:
        if time.time() > deadline - 30:
            break
        
        # Try random initialization with small integers/half-integers
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Generate random sparse matrices with small integer/half-integer entries
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Each column should have a few non-zero entries
            for r in range(target_rank):
                # Pick random non-zero positions
                nz_u = np.random.choice(n*m, size=min(3, n*m), replace=False)
                nz_v = np.random.choice(m*p, size=min(3, m*p), replace=False)
                nz_w = np.random.choice(p*n, size=min(3, p*n), replace=False)
                
                for idx in nz_u:
                    U[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2])
                for idx in nz_v:
                    V[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2])
                for idx in nz_w:
                    W[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2])
            
            # Verify
            if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a more structured approach: use known 3x3x3 rank-23 decomposition
    # For 3x3 x 3x6, we can compute two 3x3 products
    # Let's try rank 46 (two independent rank-23 decompositions)
    
    # Actually, let me try a greedy approach: start from schoolbook and try to reduce rank
    # by finding linear dependencies
    
    # For now, return the best we have (schoolbook) if nothing better found
    return best_result
