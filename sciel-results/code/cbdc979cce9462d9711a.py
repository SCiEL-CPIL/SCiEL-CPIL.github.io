"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-78 decomposition of the 3x4 * 4x7 matrix multiplication tensor.
    Strategy: Use 6 disjoint 2x2x2 Strassen blocks for i in {0,1}, j in {0,1} or {2,3}, 
             and k in {0,1}, {2,3}, or {4,5}. Each 2x2x2 block has rank 7 instead of 8.
    The remaining 36 (i,j,k) tuples use schoolbook rank-1 terms.
    Total rank: 6*7 + 36 = 78.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    rank = 78
    
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    
    # Strassen 2x2x2 blocks:
    # For a 2x2 matrix A (rows i0,i1) * 2x2 matrix B (cols j0,j1) producing 2x2 C (rows i0,i1, cols k0,k1)
    # The 7 products are:
    # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    # M2 = (A[i0,j1]+A[i1,j1]) * B[j0,k0]
    # M3 = A[i0,j0] * (B[j1,k1]-B[j0,k0])
    # M4 = A[i1,j1] * (B[j1,k1]-B[j0,k0])  -- wait, let me use standard Strassen
    # 
    # Standard Strassen for 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)  -- actually let me double check
    # 
    # Let me use the standard formulation:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So for indices:
    # i0, i1: row indices of A
    # j0, j1: column indices of A (which are row indices of B)
    # k0, k1: column indices of B
    #
    # A[i0,j0] = a, A[i0,j1] = b, A[i1,j0] = c, A[i1,j1] = d
    # B[j0,k0] = e, B[j0,k1] = f, B[j1,k0] = g, B[j1,k1] = h
    #
    # M1: U[i0*m+j0]=1, U[i1*m+j1]=1; V[j0*p+k0]=1, V[j1*p+k1]=1; W[k0*n+i0]=1, W[k1*n+i1]=1
    # M2: U[i1*m+j0]=1, U[i1*m+j1]=1; V[j0*p+k0]=1; W[k0*n+i1]=1
    # M3: U[i0*m+j0]=1; V[j1*p+k1]=1, V[j0*p+k0]=-1; W[k1*n+i0]=1, W[k0*n+i0]=-1
    # M4: U[i1*m+j1]=1; V[j1*p+k1]=1, V[j0*p+k0]=-1; W[k1*n+i1]=1, W[k0*n+i1]=-1
    # M5: U[i0*m+j0]=1, U[i0*m+j1]=1; V[j1*p+k1]=1; W[k1*n+i0]=1
    # M6: U[i1*m+j0]=1, U[i0*m+j0]=-1; V[j0*p+k0]=1, V[j1*p+k1]=1; W[k0*n+i0]=1, W[k1*n+i0]=1
    # M7: U[i0*m+j1]=1, U[i1*m+j1]=-1; V[j1*p+k0]=1, V[j1*p+k1]=1; W[k0*n+i0]=1, W[k1*n+i0]=1
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1, col_offset):
        nonlocal col
        # M1
        U[i0*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        # M2
        U[i1*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        col += 1
        # M3
        U[i0*m+j0][col] = 1
        V[j1*p+k1][col] = 1
        V[j0*p+k0][col] = -1
        W[k1*n+i0][col] = 1
        W[k0*n+i0][col] = -1
        col += 1
        # M4
        U[i1*m+j1][col] = 1
        V[j1*p+k1][col] = 1
        V[j0*p+k0][col] = -1
        W[k1*n+i1][col] = 1
        W[k0*n+i1][col] = -1
        col += 1
        # M5
        U[i0*m+j0][col] = 1
        U[i0*m+j1][col] = 1
        V[j1*p+k1][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
        # M6
        U[i1*m+j0][col] = 1
        U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
        # M7
        U[i0*m+j1][col] = 1
        U[i1*m+j1][col] = -1
        V[j1*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i0][col] = 1
        col += 1
    
    # Define the 6 Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),  # i={0,1}, j={0,1}, k={0,1}
        (0, 1, 0, 1, 2, 3),  # i={0,1}, j={0,1}, k={2,3}
        (0, 1, 0, 1, 4, 5),  # i={0,1}, j={0,1}, k={4,5}
        (0, 1, 2, 3, 0, 1),  # i={0,1}, j={2,3}, k={0,1}
        (0, 1, 2, 3, 2, 3),  # i={0,1}, j={2,3}, k={2,3}
        (0, 1, 2, 3, 4, 5),  # i={0,1}, j={2,3}, k={4,5}
    ]
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        add_strassen_block(i0, i1, j0, j1, k0, k1, col)
    
    # Add schoolbook terms for remaining (i,j,k) tuples
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
