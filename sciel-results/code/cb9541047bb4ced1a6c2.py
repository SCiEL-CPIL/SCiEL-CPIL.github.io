import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition of rank 41 using:
    # - 7 terms for the 2x2x2 sub-block (Strassen)
    # - 14 terms for the remaining 2x2x1 and 2x1x2 sub-blocks (trivial)
    # - 20 terms for the 1x3x7 and 3x1x7 sub-blocks (trivial)
    # Total = 7 + 14 + 20 = 41
    
    U = np.zeros((n * m, 41))
    V = np.zeros((m * p, 41))
    W = np.zeros((p * n, 41))
    
    col = 0
    
    # Strassen's 7 products for the 2x2x2 sub-block:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i*m + j where i in {0,1}, j in {0,1} -> {0, 1, 3, 4}
    # B indices: j*p + k where j in {0,1}, k in {0,1} -> {0, 1, 7, 8}
    # C indices: k*n + i where k in {0,1}, i in {0,1} -> {0, 1, 2, 3}
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, col] += 1; U[3, col] += 1
    V[0, col] += 1; V[7, col] += 1
    W[0, col] += 1; W[3, col] += 1
    col += 1
    
    # M2 = (A10 + A11)B00
    U[1, col] += 1; U[3, col] += 1
    V[0, col] += 1
    W[2, col] += 1
    col += 1
    
    # M3 = A00(B10 + B11)
    U[0, col] += 1
    V[7, col] += 1; V[8, col] += 1
    W[0, col] += 1
    col += 1
    
    # M4 = A11(B01 + B10)
    U[3, col] += 1
    V[1, col] += 1; V[7, col] += 1
    W[2, col] += 1; W[3, col] += 1
    col += 1
    
    # M5 = (A00 + A01)B11
    U[0, col] += 1; U[1, col] += 1
    V[8, col] += 1
    W[3, col] += 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[1, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[2, col] += 1
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, col] += 1; U[3, col] -= 1
    V[7, col] += 1; V[8, col] += 1
    W[0, col] += 1
    col += 1
    
    # Now handle the remaining terms with trivial rank-1 decompositions.
    # Total terms to add: 14 + 20 = 34
    # We iterate over all (i, j, k) that are NOT in the 2x2x2 sub-block (i<2, j<2, k<2).
    # For each such (i, j, k), we add a trivial term.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                # Trivial term: A[i,j] * B[j,k] -> C[i,k]
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # col should be 41
    assert col == 41, f"Expected 41 columns, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
