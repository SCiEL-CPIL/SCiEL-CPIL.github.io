"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-23 exact decomposition for 3x3 matrix multiplication.
    Falls back to the schoolbook rank-27 decomposition if no valid rank-23 
    solution is found within the time limit.
    """
    n, m, p = 3, 3, 3
    
    # Fallback: Schoolbook rank 27
    def get_schoolbook():
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verification function
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * 3 + j][r] * V[j * 3 + k][r] * W[k * 3 + i][r]
                    # Check if s is close to 1
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find rank 23 by starting from schoolbook and eliminating terms
    # Strategy: Use the fact that we can add linear combinations of existing terms
    # to cancel out individual terms, then remove them.
    
    start_time = time.time()
    time_limit = 110.0  # Keep 10s safety margin
    
    def try_rank_reduction(rank_target):
        """Try to find a valid decomposition of the given rank."""
        if time.time() - start_time > time_limit:
            return None
        
        # Start with schoolbook and try to merge terms
        # For rank 23, we need to eliminate 4 terms from 27
        
        # Approach: Try all subsets of 4 terms to eliminate
        # For each subset, check if the remaining 23 terms can be adjusted
        # to still satisfy all 27 equations
        
        # This is complex. Let's try a numerical approach instead.
        
        # Initialize with random values in a small range
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with small random values
            scale = 1.0
            U = np.random.randint(-2, 3, (9, rank_target)).astype(float)
            V = np.random.randint(-2, 3, (9, rank_target)).astype(float)
            W = np.random.randint(-2, 3, (9, rank_target)).astype(float)
            
            # Add half-integers
            for mat in [U, V, W]:
                mask = np.random.rand(9, rank_target) < 0.3
                mat[mask] += 0.5
            
            # Compute error
            def compute_error(U, V, W):
                err = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for r in range(rank_target):
                                s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                            err += (s - 1.0)**2
                return err
            
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            return (best_U, best_V, best_W)
        
        return None
    
    # Try rank 23
    result = try_rank_reduction(23)
    if result is not None:
        U, V, W = result
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-END
