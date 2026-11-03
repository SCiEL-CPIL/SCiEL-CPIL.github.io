"""Combined approach: block structure + search for lower-rank 3x3x3 decompositions."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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


def _verify_decomposition(U, V, W, rank, n, m, p):
    """Verify exact decomposition using the trace convention W[k*n+i, r]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_block_decomposition(n, m, p, target_rank, max_iterations=5000, seed=42):
    """
    Try to find a decomposition of rank target_rank for n x m x p matrix multiplication.
    Uses randomized initialization and iterative refinement.
    Returns (U, V, W) as lists if found, None otherwise.
    """
    rng = np.random.RandomState(seed)
    
    for iteration in range(max_iterations):
        # Initialize with random sparse matrices
        U = np.zeros((n * m, target_rank), dtype=np.float64)
        V = np.zeros((m * p, target_rank), dtype=np.float64)
        W = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Randomly assign non-zero entries
        for r in range(target_rank):
            # Pick random indices for each factor
            ui = rng.randint(0, n * m)
            vj = rng.randint(0, m * p)
            wk = rng.randint(0, p * n)
            
            # Ensure consistency: ui = i*m+j, vj = j*p+k, wk = k*n+i
            # We need to find i,j,k such that these hold
            # For now, just set random values
            U[ui, r] = rng.choice([-1, 1])
            V[vj, r] = rng.choice([-1, 1])
            W[wk, r] = rng.choice([-1, 1])
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), target_rank, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_structured_rank23():
    """
    Try to construct a known rank-23 decomposition for 3x3x3.
    This uses a specific algebraic construction.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    U = np.zeros((9, rank), dtype=np.float64)
    V = np.zeros((9, rank), dtype=np.float64)
    W = np.zeros((9, rank), dtype=np.float64)
    
    # Use a known construction based on Strassen-like ideas
    # This is a heuristic attempt; may not be valid
    
    # Term 0-8: Base terms for j=0
    for i in range(3):
        for k in range(3):
            r = i * 3 + k
            U[i * 3 + 0, r] = 1
            V[0 * 3 + k, r] = 1
            W[k * 3 + i, r] = 1
    
    # Term 9-17: Base terms for j=1
    for i in range(3):
        for k in range(3):
            r = 9 + i * 3 + k
            U[i * 3 + 1, r] = 1
            V[1 * 3 + k, r] = 1
            W[k * 3 + i, r] = 1
    
    # Term 18-22: Partial terms for j=2 (only 5 of 9 needed if rank 23)
    # This is incomplete; we need a proper construction
    
    # For now, let's use a different approach: try rank 27 (schoolbook) for each block
    # This gives total rank 54, same as original schoolbook
    
    return None  # Not a valid construction yet


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # First, try to construct a better decomposition using block structure
    # 3x3x6 can be seen as two 3x3x3 blocks
    
    # Try to find valid decompositions for 3x3x3 of various ranks
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank 54
    
    # Try rank 23 for 3x3x3 (known minimum)
    for target_rank in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 20:
            break
        
        result = _build_block_decomposition(3, 3, 3, target_rank, max_iterations=2000, seed=target_rank)
        if result is not None:
            U_block, V_block, W_block = result
            # Construct full 3x3x6 decomposition
            rank_full = 2 * target_rank
            U_full = np.zeros((9, rank_full), dtype=np.float64)
            V_full = np.zeros((18, rank_full), dtype=np.float64)
            W_full = np.zeros((18, rank_full), dtype=np.float64)
            
            # Block 1: k in {0,1,2}
            U_full[:, 0:target_rank] = U_block
            V_full[0:9, 0:target_rank] = V_block
            W_full[0:9, 0:target_rank] = W_block
            
            # Block 2: k in {3,4,5}
            U_full[:, target_rank:rank_full] = U_block
            V_full[9:18, target_rank:rank_full] = V_block
            W_full[9:18, target_rank:rank_full] = W_block
            
            # Verify
            if _verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist(), rank_full, n, m, p):
                best_result = {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
                best_rank = rank_full
                break  # Found the best possible with this approach
    
    # If no better decomposition found, use schoolbook
    if best_result is None:
        best_result = _schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
