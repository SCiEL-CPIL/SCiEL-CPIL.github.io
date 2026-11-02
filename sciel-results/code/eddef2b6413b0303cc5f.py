"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 8 matrix.
    
    The tensor T for matrix multiplication (n x m) * (m x p) has indices (i, j, k)
    where i in [0, n), j in [0, m), k in [0, p).
    T[i, j, k] = 1 if the output C[i, k] depends on A[i, j] * B[j, k].
    
    In the standard decomposition:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor can be decomposed as:
    T = sum_c U[:, c] outer V[:, c] outer W[:, c]
    
    Where:
    - U has shape (n*m, rank), indexed by (i, j)
    - V has shape (m*p, rank), indexed by (j, k)
    - W has shape (p*n, rank), indexed by (k, i)  [Note: official trace convention W[k*n+i, r]]
    
    For the standard schoolbook:
    U[i*m + j, c] = 1 when c corresponds to (i, j, k)
    V[j*p + k, c] = 1 when c corresponds to (i, j, k)
    W[k*n + i, c] = 1 when c corresponds to (i, j, k)
    
    We need to find a rank-minimal decomposition.
    
    Known results:
    - Rank of 2x3 by 3x8 matrix multiplication tensor.
    - We can try to find a decomposition with rank less than 48 (schoolbook).
    
    Strategy:
    1. Try to find known low-rank decompositions from literature.
    2. Use random search / optimization to find low-rank decompositions.
    3. Fall back to schoolbook if no better solution is found within time limit.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin before 150s kill
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 48
    
    # Try to find a better decomposition
    # Known: R(2, 3, 3) <= 11, R(2, 3, 2) <= 10
    # R(2, 3, 8) could potentially be lower than 48
    
    # Approach: Use a random search to find low-rank decompositions
    # The condition for a valid decomposition of rank r is:
    # For all (i, j, k): T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Since T[i,j,k] = 1 for all (i,j,k) in the tensor, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of equations. Let's try to find solutions with small rank.
    
    # First, let's try known decompositions from the literature
    # For 2x3 by 3x8, one approach is to use the fact that:
    # 2x3 by 3x8 can be decomposed using Kronecker products or similar
    
    # Let's try a random search approach with integer/half-integer entries
    best_result = None
    best_rank = schoolbook_rank
    
    # Create the target tensor values
    # T[i, j, k] = 1 for all i, j, k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from a lower bound up to schoolbook_rank
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # But actual lower bound for tensor rank can be higher
    
    # Let's try a structured approach first
    # For 2x3 by 3x8, we can try to exploit the structure
    
    # Method 1: Try to find a decomposition using known low-rank components
    # R(2, 3, 3) <= 11 (known)
    # R(2, 3, 2) <= 10 (known)
    # 8 = 3 + 3 + 2, so R(2, 3, 8) <= R(2, 3, 3) + R(2, 3, 3) + R(2, 3, 2) = 11 + 11 + 10 = 32
    
    # But we need the actual decompositions. Let's try to construct them.
    
    # For now, let's do a random search for low-rank solutions
    # Use rational numbers to ensure exactness
    
    random.seed(42)
    
    # Try ranks from 24 to schoolbook_rank - 1
    # Start with promising ranks
    ranks_to_try = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try to find a random solution of this rank
        # Initialize with small random integers
        attempts = 0
        max_attempts = 50 if rank <= 30 else 20
        
        while attempts < max_attempts:
            if time.time() > deadline - 10:
                break
            attempts += 1
            
            # Initialize U, V, W with small random integers
            scale = 2
            U = np.zeros((n * m, rank), dtype=object)
            V = np.zeros((m * p, rank), dtype=object)
            W = np.zeros((p * n, rank), dtype=object)
            
            # Fill with random small integers
            for i in range(n * m):
                for c in range(rank):
                    U[i, c] = random.randint(-scale, scale)
            for i in range(m * p):
                for c in range(rank):
                    V[i, c] = random.randint(-scale, scale)
            for i in range(p * n):
                for c in range(rank):
                    W[i, c] = random.randint(-scale, scale)
            
            # Check if this is a valid decomposition
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        total = 0
                        for c in range(rank):
                            total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if total != 1:
                            valid = False
                            break
            
            if valid:
                # Convert to proper format
                U_list = [[int(U[i, c]) for c in range(rank)] for i in range(n * m)]
                V_list = [[int(V[i, c]) for c in range(rank)] for i in range(m * p)]
                W_list = [[int(W[i, c]) for c in range(rank)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a valid solution with this rank, we could try lower ranks
        # But let's continue to see if we can do even better
    
    # If no better solution found, return schoolbook
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
# EVOLVE-BLOCK-END
