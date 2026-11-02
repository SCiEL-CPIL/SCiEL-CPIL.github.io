"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    n, m, p = 3, 4, 7
    deadline = time.time() + 140  # 140s deadline (10s safety margin)
    
    def schoolbook():
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
    
    def validate(U, V, W):
        """Check if decomposition is exact using integer/rational arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def vectorized_error(U, V, W):
        """Compute sum of squared errors using numpy."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We want T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # This is a bit tricky to vectorize directly without loops over i,j,k because of the index mapping.
        # However, we can reshape and use einsum.
        # U reshaped to (n, m, rank)
        # V reshaped to (m, p, rank)
        # W reshaped to (p, n, rank)
        
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # T[i, j, k] = sum_c U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c]
        # We can compute this using einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Target is all 1s
        target = np.ones((n, m, p))
        err = np.sum((T - target) ** 2)
        return err

    best_result = schoolbook()
    best_rank = n * m * p
    
    np.random.seed(42)
    
    def search_rank(rank, time_limit=10.0):
        """Search for a valid decomposition of given rank."""
        start_time = time.time()
        
        # Initialize with schoolbook-like structure but compressed?
        # Or random sparse.
        # Let's try random sparse with {-1, 0, 1}
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        # Initialize with a few random non-zeros
        for c in range(rank):
            # Each term should contribute to some (i,j,k)
            # Randomly pick i, j, k
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            U[i*m+j, c] = np.random.choice([-1, 1])
            V[j*p+k, c] = np.random.choice([-1, 1])
            W[k*n+i, c] = np.random.choice([-1, 1])
            
        best_err = vectorized_error(U, V, W)
        
        # Local Search
        while time.time() - start_time < time_limit:
            # Pick a random matrix and a random entry to flip or set
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                mat = U
                rows = n*m
            elif mat_choice == 1:
                mat = V
                rows = m*p
            else:
                mat = W
                rows = p*n
                
            r = np.random.randint(rows)
            c = np.random.randint(rank)
            
            # Try flipping or setting to 0, 1, -1
            old_val = mat[r, c]
            new_val = np.random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
                
            mat[r, c] = new_val
            
            new_err = vectorized_error(U, V, W)
            
            if new_err < best_err:
                best_err = new_err
                if best_err == 0:
                    return U, V, W
            else:
                # Revert
                mat[r, c] = old_val
                
        return None

    # Try to find better ranks
    # We know schoolbook is 84.
    # Try 83 down to 70.
    for target_rank in range(83, 69, -1):
        if time.time() > deadline - 20:
            break
            
        # Run search for this rank
        result = search_rank(target_rank, time_limit=5.0)
        if result is not None:
            U, V, W = result
            U_out = U.tolist()
            V_out = V.tolist()
            W_out = W.tolist()
            
            if validate(U_out, V_out, W_out):
                return {"U": U_out, "V": V_out, "W": W_out}
                
    return best_result
# EVOLVE-BLOCK-END
