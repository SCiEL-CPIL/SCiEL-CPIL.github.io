import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    # Matrix multiplication: C = A * B
    # A is 4x4 (n x m), B is 4x7 (m x p), C is 4x7 (n x p)
    # T[i,j,k] = 1 for all valid (i,j,k)
    #
    # Rank-1 term: u ⊗ v ⊗ w where:
    #   u indexed by (i,j) -> size n*m = 16
    #   v indexed by (j,k) -> size m*p = 28
    #   w indexed by (k,i) -> size p*n = 28
    #
    # We need: sum_c U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i] = 1 for all i,j,k
    #
    # Strategy:
    # - For the 4x4 * 4x4 part (k=0,1,2,3): use recursive Strassen (rank 49)
    # - For the 4x4 * 4x3 part (k=4,5,6): use schoolbook (rank 48)
    # Total: 49 + 48 = 97

    terms = []  # list of (u, v, w) triples

    def strassen_2x2(i_off, j_off, k_off):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block multiplication.
        i_off, j_off, k_off are the top-left offsets in the full matrices.
        """
        result = []

        # M1 = (A11+A22)*(B11+B22)
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M2 = (A21+A22)*B11
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        result.append((u[:], v[:], w[:]))

        # M3 = A11*(B12-B22)
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M4 = A22*(B21-B11)
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M5 = (A11+A12)*B22
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        result.append((u[:], v[:], w[:]))

        # M6 = (A21-A11)*(B11+B12)
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M7 = (A12-A22)*(B21+B22)
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u[:], w[:]))

        return result

    # Recursive Strassen for 4x4*4x4
    # The 4x4 blocks are:
    # A11: i_off=0, j_off=0
    # A12: i_off=0, j_off=2
    # A21: i_off=2, j_off=0
    # A22: i_off=2, j_off=2
    # B11: j_off=0, k_off=0
    # B12: j_off=0, k_off=2
    # B21: j_off=2, k_off=0
    # B22: j_off=2, k_off=2

    # For the recursive Strassen, each of the 7 products is a 2x2*2x2 where
    # the "2x2" matrices are formed by taking one element from each of the
    # four 2x2 blocks.

    # For M1 = (A11+A22)*(B11+B22):
    # The virtual A matrix has:
    #   A'[0,0] = A[0,0] + A[2,2]  (A11[0,0] + A22[0,0])
    #   A'[0,1] = A[0,1] + A[2,3]  (A11[0,1] + A22[0,1])
    #   A'[1,0] = A[1,0] + A[3,2]  (A11[1,0] + A22[1,0])
    #   A'[1,1] = A[1,1] + A[3,3]  (A11[1,1] + A22[1,1])

    # For the inner Strassen on virtual 2x2*2x2:
    # A'_11 = A'[0,0] = A[0,0]+A[2,2], A'_12 = A'[0,1] = A[0,1]+A[2,3]
    # A'_21 = A'[1,0] = A[1,0]+A[3,2], A'_22 = A'[1,1] = A[1,1]+A[3,3]
    # B'_11 = B'[0,0] = B[0,0]+B[2,2], B'_12 = B'[0,1] = B[0,1]+B[2,3]
    # B'_21 = B'[1,0] = B[1,0]+B[3,2], B'_22 = B'[1,1] = B[1,1]+B[3,3]
    # C'_11 = C'[0,0] = C[0,0], C'_12 = C'[0,1] = C[0,1]
    # C'_21 = C'[1,0] = C[1,0], C'_22 = C'[1,1] = C[1,1]

    # This is getting complex. Let me just use the 56-term approach
    # which I know works, and then add schoolbook for the remaining 3 columns.
    # Total rank: 56 + 48 = 104.

    # Actually, let me try the 49-term recursive Strassen properly.

    # For each of the 7 Strassen products, I need to generate 7 rank-1 terms.
    # Each rank-1 term in the inner Strassen corresponds to a specific combination
    # of the virtual 2x2 elements.

    # Let me define the mapping for each of the 7 outer Strassen products:

    # For product Mx, the virtual A and B matrices are linear combinations
    # of the actual 2x2 blocks. Each "element" of the virtual matrix is a
    # sum of 1 or 2 actual elements.

    # For the inner Strassen, each of the 7 terms involves products of
    # virtual elements. Each virtual element is a sum of actual elements.
    # So each inner term expands to a sum of products of actual elements.

    # But in the tensor decomposition, we can have a single rank-1 term
    # where the u-vector has non-zero entries at multiple positions (with
    # appropriate signs), representing the linear combination.

    # Let me define the 7 outer products and their virtual element mappings:

    # M1 = (A11+A22)*(B11+B22)
    # Virtual A'[r,c] = A[2r,2c] + A[2r+2,2c+2]
    # Virtual B'[r,c] = B[2r,2c] + B[2r+2,2c+2]
    # Virtual C'[r,c] = C[2r,2c]

    # For the inner Strassen on virtual 2x2*2x2:
    # A'_11 = A'[0,0] = A[0,0]+A[2,2], A'_12 = A'[0,1] = A[0,1]+A[2,3]
    # A'_21 = A'[1,0] = A[1,0]+A[3,2], A'_22 = A'[1,1] = A[1,1]+A[3,3]
    # B'_11 = B'[0,0] = B[0,0]+B[2,2], B'_12 = B'[0,1] = B[0,1]+B[2,3]
    # B'_21 = B'[1,0] = B[1,0]+B[3,2], B'_22 = B'[1,1] = B[1,1]+B[3,3]
    # C'_11 = C'[0,0] = C[0,0], C'_12 = C'[0,1] = C[0,1]
    # C'_21 = C'[1,0] = C[1,0], C'_22 = C'[1,1] = C[1,1]

    # This is getting very complex. Let me just use the 56-term approach
    # for simplicity and correctness.

    # 56-term approach: 8 separate 2x2*2x2 Strassen decompositions
    # for the 4x4*4x4 part.

    # C11 = A11*B11 + A12*B21
    terms.extend(strassen_2x2(0, 0, 0))  # A11*B11
    terms.extend(strassen_2x2(0, 2, 0))  # A12*B21

    # C12 = A11*B12 + A12*B22
    terms.extend(strassen_2x2(0, 0, 2))  # A11*B12
    terms.extend(strassen_2x2(0, 2, 2))  # A12*B22

    # C21 = A21*B11 + A22*B21
    terms.extend(strassen_2x2(2, 0, 0))  # A21*B11
    terms.extend(strassen_2x2(2, 2, 0))  # A22*B21

    # C22 = A21*B12 + A22*B22
    terms.extend(strassen_2x2(2, 0, 2))  # A21*B12
    terms.extend(strassen_2x2(2, 2, 2))  # A22*B22

    # Schoolbook for 4x4 * 4x3 (k=4,5,6)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
