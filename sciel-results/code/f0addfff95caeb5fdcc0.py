"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """Return a tensor decomposition for 2x3x9 matrix multiplication.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    The tensor T[i,j,k] = 1 for all valid (i,j,k).
    """
    start_time = time.time()
    DEADLINE = 135  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
        rank = schoolbook_rank
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
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition using integer arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        s += u_row[c] * v_row[c] * w_row[c]
                    if s != 1:
                        return False
        return True
    
    def verify_float(U, V, W, rank, tol=1e-6):
        """Fast float verification for screening."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Reshape for tensor contraction
        # U: (n*m, rank) -> (n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        
        U_t = U_np.reshape(n, m, rank)
        V_t = V_np.reshape(m, p, rank)
        W_t = W_np.reshape(p, n, rank)
        
        # For each (i, j, k), check sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1
        # This is a 3-tensor contraction.
        # Let's just loop for clarity and speed (small dimensions)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.dot(U_t[i, j, :], V_t[j, k, :]), W_t[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def random_search(rank, num_trials, seed=None):
        """Attempt to find a rank-rank decomposition via random search."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        entries_pool = [-2, -1, 0, 0, 0, 1, 1, 2]
        
        for t in range(num_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Generate random matrices
            U = [[random.choice(entries_pool) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(entries_pool) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(entries_pool) for _ in range(rank)] for _ in range(p * n)]
            
            # Quick float check
            if verify_float(U, V, W, rank):
                # Exact check
                if verify_exact(U, V, W, rank):
                    return {"U": U, "V": V, "W": W}
        return None

    def structured_search(rank):
        """
        Try to construct a decomposition using block structure.
        The matrix multiplication tensor for (2,3,9) can be viewed as:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        We can try to exploit the small size of n=2 and m=3.
        
        One idea: Use a decomposition that respects the structure.
        For example, if we could find rank 27 (the dimension of V),
        we would need each column of W to be a linear combination.
        
        However, random search is weak here. Let's try a more targeted approach.
        Since n=2 is small, we can think of the tensor as two 3x9 matrices being multiplied.
        
        Actually, let's try a "lifting" approach or just stick to random search
        with more trials for lower ranks, as structured construction is hard without
        known algorithms for this specific shape.
        
        Let's try a hybrid: Start with schoolbook, then try to remove columns
        by solving linear systems for the remaining constraints? No, that's complex.
        
        Let's focus on optimizing the random search to be faster and try
        a few specific ranks that might be plausible.
        """
        pass

    # Strategy:
    # 1. Try random search for ranks from 40 down to 30 (just below schoolbook).
    # 2. If found, return it.
    # 3. If not, return schoolbook.
    
    # Theoretical lower bound is max(n*p, n*m, m*p) = max(18, 6, 27) = 27.
    # But finding a rank 27-40 decomposition is very hard with random search.
    # Let's try ranks 53, 52, ... down to 45 with a few trials each.
    
    best_result = None
    
    # We will iterate over target ranks from 53 down to 40.
    # For each rank, we do a limited number of random trials.
    # The total time budget is ~135s.
    
    # Let's estimate time per trial.
    # verify_float uses numpy, should be fast.
    # verify_exact is pure python, slower.
    # Most trials will fail verify_float quickly.
    
    ranks_to_try = range(53, 39, -1)  # 53 down to 40
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Number of trials depends on remaining time
        # Let's do a fixed number of trials per rank, say 500, but check time.
        trials = 500
        
        result = random_search(target_rank, trials)
        if result is not None:
            best_result = result
            print(f"Found rank {target_rank} decomposition!")
            return best_result

    # If no better decomposition found, return schoolbook
    return build_schoolbook()
# EVOLVE-BLOCK-END
