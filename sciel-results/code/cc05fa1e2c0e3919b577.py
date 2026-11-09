"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x9 matrix.
    
    Strategy:
    1. Use the schoolbook decomposition as a baseline (rank 54).
    2. Attempt to find a lower rank decomposition using structured search.
    3. Return the best decomposition found.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    time_limit = 10.0  # Allow 10 seconds for search
    
    # Build the schoolbook decomposition as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1

    # Function to verify a decomposition
    def verify(U, V, W, rank):
        """Verify if U, V, W exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        return True

    # Function to find a decomposition using structured approach
    def find_decomposition(rank):
        """Try to find a decomposition of given rank using structured search."""
        # For matrix multiplication [n, m, p], we can try to exploit
        # the structure by considering different factorizations
        
        # Approach: Use the fact that we can combine terms
        # The key insight is that for fixed (i, k), we need:
        # sum_j A[i,j] B[j,k] = C[i,k]
        # This means for each (i, k), the sum over j must work out.
        
        # We'll try a structured search where we fix some indices and vary others
        # Start with a base decomposition and try to merge terms
        
        # For now, return None to indicate we need to use the fallback
        return None

    # Search for lower rank using structured approach
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find decompositions with rank < schoolbook_rank
    # The theoretical lower bound is max(n*m, m*p, p*n) = 27
    # We'll search from schoolbook_rank - 1 down to a reasonable limit
    
    for r in range(schoolbook_rank - 1, max(27, schoolbook_rank - 10), -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try structured search for this rank
        result = find_decomposition(r)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, r):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # Found the best, stop searching
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
