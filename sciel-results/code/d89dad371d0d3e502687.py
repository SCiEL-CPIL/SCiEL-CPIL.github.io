"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Use a rank-105 decomposition by exploiting linear dependencies
    # The idea: for the last column of A (j=3), we can share computations
    # with the other columns in a structured way.
    
    rank = 105
    U = np.zeros((n * m, rank), dtype=float)  # 16 x 105
    V = np.zeros((m * p, rank), dtype=float)  # 28 x 105
    W = np.zeros((p * n, rank), dtype=float)  # 28 x 105
    
    col = 0
    # For j = 0, 1, 2: use standard schoolbook terms (3 * 4 * 7 = 84 terms)
    for i in range(n):
        for j in range(3):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # For j = 3: use a reduced decomposition
    # We need to compute sum_{i,k} A[i,3]*B[3,k] * W
    # But we can potentially share with other terms.
    # 
    # Actually, let me think more carefully. The standard decomposition has
    # 4*4*7 = 112 terms. To reduce to 105, we need to eliminate 7 terms.
    #
    # One approach: for each fixed i and k, we have 4 terms (one for each j).
    # If we can express one of these as a combination of the others, we can reduce.
    #
    # Let's try a different approach: use a bilinear form that shares structure.
    # For the last 7 terms (one for each k when i=0, j=3), we can try to
    # express them using the existing terms.
    
    # Reset and use a proper approach
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Standard terms for all (i, j, k) except we'll handle j=3 differently
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == 3 and k == 0:
                    continue  # Skip this one, we'll handle it differently
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now we have used 111 terms. We need to express the missing term
    # A[i,3]*B[3,0] for all i in terms of the others.
    # This isn't straightforward. Let me use the full schoolbook.
    
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
    
    # Convert to lists for the return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
