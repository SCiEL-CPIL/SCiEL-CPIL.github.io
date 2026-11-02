"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly matches the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 for all valid (i,j,k) and 0 otherwise (but since we only
    have valid indices, it should be 1 for all).
    """
    rank = len(U[0]) if U else 0
    # Build the tensor via the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
    # (since C[i,k] = sum_j A[i,j]*B[j,k], the tensor has 1s everywhere)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_strassen_based(n, m, p):
    """
    Try to construct a decomposition based on Strassen's algorithm for 3x3x3
    extended to 3x3x5.
    
    Strassen gives rank 7 for 2x2x2, and can be adapted.
    For 3x3x3, the known minimal rank is 19 (Landsberg) or 23 (schoolbook).
    Actually, the best known upper bound for 3x3x3 is 19 (Landsberg 1987),
    but constructing it is complex.
    
    A simpler approach: use the schoolbook for the first 3 columns of B,
    then add terms for the remaining 2 columns.
    
    Schoolbook for 3x3x3: rank 27
    Then for each additional column of B (2 more columns), we need 3x3=9 more terms.
    Total: 27 + 2*9 = 45. That's just schoolbook again.
    
    Alternative: use a block approach.
    Split B into B1 (3x3) and B2 (3x2).
    C = A*B1 + A*B2
    
    For A*B1, use Strassen-like or schoolbook (rank 27 for schoolbook).
    For A*B2, use schoolbook (rank 9).
    Total: 36.
    
    But can we do better? Let's try to share some terms.
    
    Actually, let's just try a direct search for rank 36 using the structure.
    """
    pass  # We'll implement a concrete construction below

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s kill
    
    # First, get the schoolbook decomposition as a fallback
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # Strategy: Use the fact that for 3x3x5, we can potentially do better than 45
    # by exploiting the structure.
    
    # Let's try a direct construction:
    # Split the 5 columns of B into 3 + 2.
    # For the first 3 columns, use a rank-19 decomposition of 3x3x3 (if we can construct it).
    # For the last 2 columns, use rank 9 (schoolbook for 3x3x2).
    # Total: 19 + 9 = 28.
    
    # However, constructing the rank-19 decomposition of 3x3x3 is non-trivial.
    # Let's try a simpler approach: use schoolbook for the whole thing but see
    # if we can merge some terms.
    
    # Actually, let's try a randomized search for a lower rank decomposition.
    # We'll try ranks from 36 down to 30, and for each rank, try to find a valid decomposition.
    
    # For efficiency, let's use a structured approach:
    # The tensor for 3x3x5 matrix multiplication has entries T[i,j,k] = 1 for all i,j,k.
    # We need U (9 x r), V (15 x r), W (15 x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Let's try to construct a decomposition with rank 36:
    # Use schoolbook for the first 4 columns (rank 36 = 3*3*4)
    # and schoolbook for the 5th column (rank 9)
    # Total: 36 + 9 = 45. Not helpful.
    
    # Let's try a different approach: use the fact that we can share some U and V entries.
    
    # For now, let's just return the schoolbook decomposition.
    # A more sophisticated search would be needed to find a better rank.
    
    return best_result

# EVOLVE-BLOCK-END
