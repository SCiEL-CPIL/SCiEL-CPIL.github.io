"""Program for 3x3x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # For 3x3x3, use a rank-23 decomposition (BSC-style)
    # For the remaining 2 columns, add 6 more terms (3x3x2 schoolbook)
    # Total rank = 23 + 6 = 29
    
    # BSC 3x3x3 decomposition (rank 23)
    # U: 9x23, V: 9x23, W: 9x23
    # This is a known valid rank-23 decomposition for 3x3 matrix multiplication
    
    # The BSC decomposition uses 23 bilinear forms.
    # Let me use a specific valid set of coefficients.
    
    # Standard BSC coefficients for 3x3x3 rank 23:
    # Using the decomposition from Bini, Strassen, Caporaso (1980)
    
    # U[i*m+j, r] for i,j in 0..2, r in 0..22
    U_bsc = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    # This is just the identity - not a valid BSC. Let me use actual BSC coefficients.
    # The actual BSC decomposition is more complex. Let me use a verified set.
    
    # Actually, let me just use the schoolbook decomposition for correctness,
    # but try to be smarter about it.
    
    # Schoolbook for 3x3x5: rank = 3*3*5 = 45
    
    # But we can do better by using the structure.
    # For each output column k, we need a 3x3x3 tensor.
    # If we use the same decomposition for all 5 columns, we get rank 5*23 = 115 (worse).
    # If we use schoolbook for each column: 5*9 = 45.
    
    # The key insight: we can share some bilinear forms across columns.
    # Specifically, if A is 3x3 and B is 3x5, then C = A*B is 3x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] has entries that are 1 if we're computing C[i,k] using A[i,j] and B[j,k].
    # For each (i,k), T[i,j,k] = 1 for all j (in the schoolbook sense, but actually each (i,j,k) is a separate term).
    
    # Let me just construct the schoolbook decomposition correctly.
    
    def schoolbook_3x3x5():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook_3x3x5()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
