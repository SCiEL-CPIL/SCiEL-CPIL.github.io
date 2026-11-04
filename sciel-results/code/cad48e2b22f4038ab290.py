"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_sparse_random_rank(target_rank, max_attempts, value_range):
        """
        Try to find a decomposition using sparse random initialization.
        Sparsity helps satisfy integer constraints more easily.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate sparse matrices
            # Each column has only a few non-zero entries
            nnz_per_col = 2  # Low sparsity for low rank
            
            U = np.zeros((nm, target_rank), dtype=np.int64)
            V = np.zeros((mp, target_rank), dtype=np.int64)
            W = np.zeros((pn, target_rank), dtype=np.int64)
            
            for c in range(target_rank):
                # Pick random rows to be non-zero
                u_rows = random.sample(range(nm), min(nnz_per_col, nm))
                v_rows = random.sample(range(mp), min(nnz_per_col, mp))
                w_rows = random.sample(range(pn), min(nnz_per_col, pn))
                
                for r in u_rows:
                    U[r, c] = random.randint(value_range[0], value_range[1])
                for r in v_rows:
                    V[r, c] = random.randint(value_range[0], value_range[1])
                for r in w_rows:
                    W[r, c] = random.randint(value_range[0], value_range[1])
            
            if verify_fast(U.tolist(), V.tolist(), W.tolist(), target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Strategy:
    # 1. Try to find a rank 21 solution (theoretical lower bound)
    # 2. If not found, try rank 22, 23, ... up to 41
    # 3. Use sparse random search which is more likely to hit exact integer solutions
    
    # We know rank 42 is always possible.
    # Let's focus on 21.
    
    # Increase attempts for lower ranks as they are harder
    # But keep total time within limits
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Adjust attempts based on rank
        # Lower rank -> harder -> more attempts needed, but time is limited
        # Let's allocate time proportionally or just fixed chunks
        # 120s / 21 ranks ~ 5s per rank? Too much.
        # Let's do 200 attempts for rank 21, 100 for others.
        
        max_attempts = 300 if target_rank == 21 else 100
        
        # Try sparse random search
        result = try_sparse_random_rank(target_rank, max_attempts, (-2, 2))
        
        if result:
            best = result
            best_rank = target_rank
            # If we found 21, we can stop as it's the lower bound
            if target_rank == 21:
                break
    
    return best
# EVOLVE-BLOCK-END
