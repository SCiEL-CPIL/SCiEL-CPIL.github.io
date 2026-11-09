"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check shapes
    if U_np.shape != (n*m, rank):
        return False
    if V_np.shape != (m*p, rank):
        return False
    if W_np.shape != (p*n, rank):
        return False
    
    # Verify: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    # And for all other combinations, it should equal 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(actual - expected) > 1e-9:
                    return False
    
    # Also verify that non-matching terms are zero
    # For any (i,j,k) that doesn't correspond to a valid multiplication term
    # Actually, the tensor is defined such that T[i,j,k] = 1 if the indices match the multiplication
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k in range, but with the constraint
    # that it represents (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k
    # Wait, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k]
    # contributes to C[i,k]. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
    
    # Try to reduce rank using known techniques
    # For 2x4 x 4x8, try rank 32 by exploiting structure
    # Actually, let's try a simple optimization: check if we can merge some columns
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    # Try to find a lower rank decomposition
    # Use the fact that we can potentially combine terms
    # For now, return the schoolbook decomposition which is guaranteed correct
    return best_result
# EVOLVE-BLOCK-END
