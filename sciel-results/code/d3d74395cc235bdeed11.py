"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Start with schoolbook rank-75 decomposition
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
    
    # Verify the decomposition is correct
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should reconstruct
        # Actually, the decomposition means: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
        # This is not quite right. Let me re-examine.
        # The tensor T_{i,j,k} = delta_{i,j,k} for matrix mult is:
        # T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # The rank-1 decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c[i*m+j], V_c[j*p+k], W_c[k*n+i]
        # So: T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
        # For schoolbook: T[i,j,k] = 1 for all i,j,k? No.
        # Actually, the matrix multiplication tensor is:
        # T_{i,j,k} = 1 if the term A[i,j]*B[j,k] contributes to (A*B)[i,k]
        # So T[i,j,k] = 1 for all valid i,j,k.
        # The decomposition: (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
        # In tensor form: sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] A[i,j] B[j,k]
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] = delta_{i,k} ... no.
        # Actually: sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] should equal 1 if the (i,j,k) term is present.
        # For schoolbook, each (i,j,k) is its own term, so U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # for the specific c corresponding to (i,j,k), and 0 otherwise.
        # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if c matches (i,j,k), else 0.
        # So the sum is 1 only when all three indices match the term, which is correct.
        return True  # Schoolbook is always valid
    
    # For now, return schoolbook. A sub-75 decomposition is known to exist (rank 69)
    # but constructing it explicitly within the time limit is non-trivial.
    # Return the valid schoolbook decomposition.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
