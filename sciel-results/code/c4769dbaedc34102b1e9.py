import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit

    # We need to decompose the matrix multiplication tensor for 4x4 * 4x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where A is 4x4, B is 4x7, C is 4x7
    #
    # The tensor has entries T[i,j,k] = 1 for all valid (i,j,k)
    # A rank-1 decomposition term is u ⊗ v ⊗ w where:
    #   u is indexed by (i,j) pairs (size n*m = 16)
    #   v is indexed by (j,k) pairs (size m*p = 28)  
    #   w is indexed by (k,i) pairs (size p*n = 28)
    #
    # We need: sum_c U[c,i,j] * V[c,j,k] * W[c,k,i] = 1 for all i,j,k
    #
    # Strategy: Use recursive Strassen for the 4x4 * 4x4 part (rank 49)
    # and schoolbook for the 4x4 * 4x3 part (rank 48)
    # Total rank: 49 + 48 = 97

    terms = []

    def strassen_2x2(i_off, j_off, k_off):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block multiplication.
        i_off, j_off, k_off are the top-left offsets in the full matrices.
        The 2x2 blocks span:
          A: rows [i_off, i_off+1], cols [j_off, j_off+1]
          B: rows [j_off, j_off+1], cols [k_off, k_off+1]
          C: rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        result = []

        # M1 = (A11+A22)*(B11+B22)
        # A11 at (i_off, j_off), A22 at (i_off+1, j_off+1)
        # B11 at (j_off, k_off), B22 at (j_off+1, k_off+1)
        # C11 += M1, C22 += M1
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
        # A21 at (i_off+1, j_off), A22 at (i_off+1, j_off+1)
        # B11 at (j_off, k_off)
        # C21 += M2, C22 -= M2
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
        # A11 at (i_off, j_off)
        # B12 at (j_off, k_off+1), B22 at (j_off+1, k_off+1)
        # C12 += M3, C22 += M3
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
        # A22 at (i_off+1, j_off+1)
        # B21 at (j_off+1, k_off), B11 at (j_off, k_off)
        # C11 += M4, C21 += M4
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
        # A11 at (i_off, j_off), A12 at (i_off, j_off+1)
        # B22 at (j_off+1, k_off+1)
        # C12 += M5, C11 -= M5
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
        # A21 at (i_off+1, j_off), A11 at (i_off, j_off)
        # B11 at (j_off, k_off), B12 at (j_off, k_off+1)
        # C22 += M6
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
        # A12 at (i_off, j_off+1), A22 at (i_off+1, j_off+1)
        # B21 at (j_off+1, k_off), B22 at (j_off+1, k_off+1)
        # C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u[:], v[:], w[:]))

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

    # For M1 = (A11+A22)*(B11+B22):
    # This is a 2x2*2x2 multiplication where the "A" matrix has entries:
    #   A'[r,c] = A[2r, 2c] + A[2r+2, 2c+2] for r,c in {0,1}
    # and the "B" matrix has entries:
    #   B'[r,c] = B[2r, 2c] + B[2r+2, 2c+2] for r,c in {0,1}
    #
    # In the tensor decomposition, we can express this by having the u-vector
    # have non-zero entries at both the A11 and A22 positions, and similarly for v.
    # The w-vector will have entries at the C11 and C22 positions.

    # Let me implement a helper that generates Strassen terms for a "virtual" 2x2*2x2
    # where the input blocks are linear combinations of the original blocks.

    def strassen_virtual(a_indices, b_indices, c_indices):
        """
        Generate 7 Strassen terms for a virtual 2x2*2x2 multiplication.
        a_indices: list of 4 (i,j) positions for the A blocks (A11, A12, A21, A22)
        b_indices: list of 4 (j,k) positions for the B blocks (B11, B12, B21, B22)
        c_indices: list of 4 (k,i) positions for the C blocks (C11, C12, C21, C22)
        
        Each position is a tuple (row, col) in the full matrix.
        """
        result = []
        
        # A11, A12, A21, A22
        a11, a12, a21, a22 = a_indices
        # B11, B12, B21, B22
        b11, b12, b21, b22 = b_indices
        # C11, C12, C21, C22
        c11, c12, c21, c22 = c_indices

        # M1 = (A11+A22)*(B11+B22)
        # Contributes: C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[a11[0] * m + a11[1]] = 1
        u[a22[0] * m + a22[1]] = 1
        v = [0] * (m * p)
        v[b11[0] * p + b11[1]] = 1
        v[b22[0] * p + b22[1]] = 1
        w = [0] * (p * n)
        w[c11[0] * n + c11[1]] = 1
        w[c22[0] * n + c22[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M2 = (A21+A22)*B11
        # Contributes: C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[a21[0] * m + a21[1]] = 1
        u[a22[0] * m + a22[1]] = 1
        v = [0] * (m * p)
        v[b11[0] * p + b11[1]] = 1
        w = [0] * (p * n)
        w[c21[0] * n + c21[1]] = 1
        w[c22[0] * n + c22[1]] = -1
        result.append((u[:], v[:], w[:]))

        # M3 = A11*(B12-B22)
        # Contributes: C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[a11[0] * m + a11[1]] = 1
        v = [0] * (m * p)
        v[b12[0] * p + b12[1]] = 1
        v[b22[0] * p + b22[1]] = -1
        w = [0] * (p * n)
        w[c12[0] * n + c12[1]] = 1
        w[c22[0] * n + c22[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M4 = A22*(B21-B11)
        # Contributes: C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[a22[0] * m + a22[1]] = 1
        v = [0] * (m * p)
        v[b21[0] * p + b21[1]] = 1
        v[b11[0] * p + b11[1]] = -1
        w = [0] * (p * n)
        w[c11[0] * n + c11[1]] = 1
        w[c21[0] * n + c21[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M5 = (A11+A12)*B22
        # Contributes: C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[a11[0] * m + a11[1]] = 1
        u[a12[0] * m + a12[1]] = 1
        v = [0] * (m * p)
        v[b22[0] * p + b22[1]] = 1
        w = [0] * (p * n)
        w[c12[0] * n + c12[1]] = 1
        w[c11[0] * n + c11[1]] = -1
        result.append((u[:], v[:], w[:]))

        # M6 = (A21-A11)*(B11+B12)
        # Contributes: C22 += M6
        u = [0] * (n * m)
        u[a21[0] * m + a21[1]] = 1
        u[a11[0] * m + a11[1]] = -1
        v = [0] * (m * p)
        v[b11[0] * p + b11[1]] = 1
        v[b12[0] * p + b12[1]] = 1
        w = [0] * (p * n)
        w[c22[0] * n + c22[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M7 = (A12-A22)*(B21+B22)
        # Contributes: C11 += M7
        u = [0] * (n * m)
        u[a12[0] * m + a12[1]] = 1
        u[a22[0] * m + a22[1]] = -1
        v = [0] * (m * p)
        v[b21[0] * p + b21[1]] = 1
        v[b22[0] * p + b22[1]] = 1
        w = [0] * (p * n)
        w[c11[0] * n + c11[1]] = 1
        result.append((u[:], v[:], w[:]))

        return result

    # For the 4x4*4x4 part, we use recursive Strassen.
    # The 7 Strassen products are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)

    # Each of these is a 2x2*2x2 multiplication. We decompose each using
    # the 7-term Strassen for 2x2*2x2.

    # For M1 = (A11+A22)*(B11+B22):
    # The A matrix has entries from A11 and A22 (both with coefficient 1)
    # The B matrix has entries from B11 and B22 (both with coefficient 1)
    # This is equivalent to a 2x2*2x2 where:
    #   A'[r,c] = A[2r, 2c] + A[2r+2, 2c+2]
    #   B'[r,c] = B[2r, 2c] + B[2r+2, 2c+2]
    #
    # In the tensor decomposition, we can express this by having the u-vector
    # have non-zero entries at both the A11 and A22 positions.

    # Let me define the positions for each Strassen product:

    # M1: A' = A11+A22, B' = B11+B22
    # A11 positions: (0,0), (0,1), (1,0), (1,1)
    # A22 positions: (2,2), (2,3), (3,2), (3,3)
    # B11 positions: (0,0), (0,1), (1,0), (1,1)
    # B22 positions: (2,2), (2,3), (3,2), (3,3)
    # C11 positions: (0,0), (0,1), (1,0), (1,1)
    # C22 positions: (2,2), (2,3), (3,2), (3,3)

    # For M1, the "virtual" 2x2*2x2 has:
    # A11_virtual = A[0,0] + A[2,2] -> position (0,0) in virtual, but in full matrix it's (0,0) and (2,2)
    # A12_virtual = A[0,1] + A[2,3] -> position (0,1) in virtual, but in full matrix it's (0,1) and (2,3)
    # A21_virtual = A[1,0] + A[3,2] -> position (1,0) in virtual, but in full matrix it's (1,0) and (3,2)
    # A22_virtual = A[1,1] + A[3,3] -> position (1,1) in virtual, but in full matrix it's (1,1) and (3,3)

    # Similarly for B and C.

    # Let me create a helper that generates the Strassen terms for a virtual 2x2*2x2
    # where each entry is a sum of two entries from the original matrices.

    def strassen_summed(a_positions, b_positions, c_positions):
        """
        Generate 7 Strassen terms for a virtual 2x2*2x2 where each entry
        is a sum of two entries from the original matrices.
        
        a_positions: list of 4 tuples, each tuple is a list of 2 (i,j) positions
        b_positions: list of 4 tuples, each tuple is a list of 2 (j,k) positions
        c_positions: list of 4 tuples, each tuple is a list of 2 (k,i) positions
        """
        result = []
        
        # A11, A12, A21, A22 - each is a list of 2 positions
        a11, a12, a21, a22 = a_positions
        # B11, B12, B21, B22
        b11, b12, b21, b22 = b_positions
        # C11, C12, C21, C22
        c11, c12, c21, c22 = c_positions

        # M1 = (A11+A22)*(B11+B22)
        # Contributes: C11 += M1, C22 += M1
        u = [0] * (n * m)
        for pos in a11:
            u[pos[0] * m + pos[1]] = 1
        for pos in a22:
            u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b11:
            v[pos[0] * p + pos[1]] = 1
        for pos in b22:
            v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c11:
            w[pos[0] * n + pos[1]] = 1
        for pos in c22:
            w[pos[0] * n + pos[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M2 = (A21+A22)*B11
        # Contributes: C21 += M2, C22 -= M2
        u = [0] * (n * m)
        for pos in a21:
            u[pos[0] * m + pos[1]] = 1
        for pos in a22:
            u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b11:
            v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c21:
            w[pos[0] * n + pos[1]] = 1
        for pos in c22:
            w[pos[0] * n + pos[1]] = -1
        result.append((u[:], v[:], w[:]))

        # M3 = A11*(B12-B22)
        # Contributes: C12 += M3, C22 += M3
        u = [0] * (n * m)
        for pos in a11:
            u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b12:
            v[pos[0] * p + pos[1]] = 1
        for pos in b22:
            v[pos[0] * p + pos[1]] = -1
        w = [0] * (p * n)
        for pos in c12:
            w[pos[0] * n + pos[1]] = 1
        for pos in c22:
            w[pos[0] * n + pos[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M4 = A22*(B21-B11)
        # Contributes: C11 += M4, C21 += M4
        u = [0] * (n * m)
        for pos in a22:
            u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b21:
            v[pos[0] * p + pos[1]] = 1
        for pos in b11:
            v[pos[0] * p + pos[1]] = -1
        w = [0] * (p * n)
        for pos in c11:
            w[pos[0] * n + pos[1]] = 1
        for pos in c21:
            w[pos[0] * n + pos[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M5 = (A11+A12)*B22
        # Contributes: C12 += M5, C11 -= M5
        u = [0] * (n * m)
        for pos in a11:
            u[pos[0] * m + pos[1]] = 1
        for pos in a12:
            u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b22:
            v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c12:
            w[pos[0] * n + pos[1]] = 1
        for pos in c11:
            w[pos[0] * n + pos[1]] = -1
        result.append((u[:], v[:], w[:]))

        # M6 = (A21-A11)*(B11+B12)
        # Contributes: C22 += M6
        u = [0] * (n * m)
        for pos in a21:
            u[pos[0] * m + pos[1]] = 1
        for pos in a11:
            u[pos[0] * m + pos[1]] = -1
        v = [0] * (m * p)
        for pos in b11:
            v[pos[0] * p + pos[1]] = 1
        for pos in b12:
            v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c22:
            w[pos[0] * n + pos[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M7 = (A12-A22)*(B21+B22)
        # Contributes: C11 += M7
        u = [0] * (n * m)
        for pos in a12:
            u[pos[0] * m + pos[1]] = 1
        for pos in a22:
            u[pos[0] * m + pos[1]] = -1
        v = [0] * (m * p)
        for pos in b21:
            v[pos[0] * p + pos[1]] = 1
        for pos in b22:
            v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c11:
            w[pos[0] * n + pos[1]] = 1
        result.append((u[:], v[:], w[:]))

        return result

    # Now let's apply this to each of the 7 Strassen products for 4x4*4x4.

    # M1 = (A11+A22)*(B11+B22)
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    # B11: (0,0), (0,1), (1,0), (1,1)
    # B22: (2,2), (2,3), (3,2), (3,3)
    # C11: (0,0), (0,1), (1,0), (1,1)
    # C22: (2,2), (2,3), (3,2), (3,3)

    # For M1, the virtual 2x2*2x2 has:
    # A11_virtual: A[0,0] + A[2,2] -> positions [(0,0), (2,2)]
    # A12_virtual: A[0,1] + A[2,3] -> positions [(0,1), (2,3)]
    # A21_virtual: A[1,0] + A[3,2] -> positions [(1,0), (3,2)]
    # A22_virtual: A[1,1] + A[3,3] -> positions [(1,1), (3,3)]

    a1_pos = [[(0,0), (2,2)], [(0,1), (2,3)], [(1,0), (3,2)], [(1,1), (3,3)]]
    b1_pos = [[(0,0), (2,2)], [(0,1), (2,3)], [(1,0), (3,2)], [(1,1), (3,3)]]
    c1_pos = [[(0,0), (2,2)], [(0,1), (2,3)], [(1,0), (3,2)], [(1,1), (3,3)]]

    terms.extend(strassen_summed(a1_pos, b1_pos, c1_pos))

    # M2 = (A21+A22)*B11
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    # B11: (0,0), (0,1), (1,0), (1,1)
    # C21: (2,0), (2,1), (3,0), (3,1)
    # C22: (2,2), (2,3), (3,2), (3,3)

    # For M2, the virtual 2x2*2x2 has:
    # A21_virtual: A[2,0] + A[2,2] -> positions [(2,0), (2,2)]
    # A22_virtual: A[2,1] + A[2,3] -> positions [(2,1), (2,3)]
    # Wait, I need to be more careful. Let me reconsider.

    # Actually, for M2 = (A21+A22)*B11, the A matrix is:
    # A'[r,c] = A[2+r, 2c] + A[2+r, 2c+2] for r,c in {0,1}
    # So:
    # A'[0,0] = A[2,0] + A[2,2]
    # A'[0,1] = A[2,1] + A[2,3]
    # A'[1,0] = A[3,0] + A[3,2]
    # A'[1,1] = A[3,1] + A[3,3]

    # And B11 is just B[0:2, 0:2], so:
    # B'[0,0] = B[0,0]
    # B'[0,1] = B[0,1]
    # B'[1,0] = B[1,0]
    # B'[1,1] = B[1,1]

    # And the output is:
    # C'[0,0] = C[2,0]
    # C'[0,1] = C[2,1]
    # C'[1,0] = C[3,0]
    # C'[1,1] = C[3,1]

    # But wait, the output of M2 contributes to C21 and C22, not just C21.
    # Let me re-examine the Strassen formulas.

    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # So M2 contributes to C21 with +1 and to C22 with -1.

    # For M2, the output C' is a 2x2 matrix where:
    # C'[r,c] = C[2+r, 2c] for the C21 part, and C[2+r, 2c+2] for the C22 part.
    # But that's not quite right either.

    # Let me think about this differently. The Strassen algorithm computes:
    # C = A * B where A and B are 4x4 matrices.
    # We split A and B into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]

    # The 7 products are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)

    # And:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each M is a 2x2*2x2 multiplication. So for M2 = (A21+A22)*B11:
    # The A matrix is A21+A22, which is a 2x2 matrix where:
    # (A21+A22)[r,c] = A21[r,c] + A22[r,c] = A[2+r, c] + A[2+r, 2+c] for r,c in {0,1}
    # So:
    # (A21+A22)[0,0] = A[2,0] + A[2,2]
    # (A21+A22)[0,1] = A[2,1] + A[2,3]
    # (A21+A22)[1,0] = A[3,0] + A[3,2]
    # (A21+A22)[1,1] = A[3,1] + A[3,3]

    # The B matrix is B11, which is a 2x2 matrix where:
    # B11[r,c] = B[2r, 2c] for r,c in {0,1}
    # So:
    # B11[0,0] = B[0,0]
    # B11[0,1] = B[0,1]
    # B11[1,0] = B[1,0]
    # B11[1,1] = B[1,1]

    # The output is a 2x2 matrix C' where:
    # C'[r,c] = C[2+r, 2c] for the C21 part (since C21 is at rows 2-3, cols 0-1)
    # But wait, C21 is a 2x2 matrix at rows 2-3, cols 0-1.
    # So C'[r,c] = C[2+r, 2c] for r,c in {0,1}.

    # But M2 also contributes to C22 with -1. C22 is at rows 2-3, cols 2-3.
    # So the output of M2 contributes to both C21 and C22.

    # In the tensor decomposition, the w-vector for M2 should have:
    # +1 at the C21 positions and -1 at the C22 positions.

    # So for M2:
    # A positions: [(2,0), (2,2)], [(2,1), (2,3)], [(3,0), (3,2)], [(3,1), (3,3)]
    # B positions: [(0,0)], [(0,1)], [(1,0)], [(1,1)]
    # C positions: [(2,0), (2,2)], [(2,1), (2,3)], [(3,0), (3,2)], [(3,1), (3,3)]

    # But wait, that's not right either. The output of M2 is a 2x2 matrix,
    # and it contributes to C21 (which is at rows 2-3, cols 0-1) and C22 (which is at rows 2-3, cols 2-3).

    # Let me reconsider. The Strassen algorithm computes C = A*B where A and B are 4x4.
    # The output C is a 4x4 matrix. The 7 products M1-M7 are each 2x2 matrices.
    # The final C is assembled from these 2x2 matrices.

    # For M2, the output is a 2x2 matrix that contributes to C21 and C22.
    # C21 is at rows 2-3, cols 0-1.
    # C22 is at rows 2-3, cols 2-3.

    # So the output of M2 is a 2x2 matrix where:
    # M2[r,c] contributes to C[2+r, 2c] (for C21) and C[2+r, 2c+2] (for C22).

    # But that's not quite right. Let me think about it more carefully.

    # Actually, I think I'm overcomplicating this. Let me just use the 56-term approach
    # (8 separate 2x2*2x2 Strassen decompositions) which is simpler and still gives
    # a valid decomposition.

    # For now, let me just use the 56-term approach for the 4x4*4x4 part.

    # Clear the terms and use the 56-term approach
    terms = []

    # Apply Strassen to each 2x2*2x2 block in the 4x4*4x4 multiplication
    # The 4x4 matrix is divided into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Each of these is a sum of two 2x2*2x2 multiplications.
    # We can apply Strassen to each 2x2*2x2 multiplication.

    # For C11 = A11*B11 + A12*B21:
    # A11*B11: use Strassen with i_off=0, j_off=0, k_off=0
    terms.extend(strassen_2x2(0, 0, 0))
    # A12*B21: use Strassen with i_off=0, j_off=2, k_off=0
    terms.extend(strassen_2x2(0, 2, 0))

    # For C12 = A11*B12 + A12*B22:
    # A11*B12: use Strassen with i_off=0, j_off=0, k_off=2
    terms.extend(strassen_2x2(0, 0, 2))
    # A12*B22: use Strassen with i_off=0, j_off=2, k_off=2
    terms.extend(strassen_2x2(0, 2, 2))

    # For C21 = A21*B11 + A22*B21:
    # A21*B11: use Strassen with i_off=2, j_off=0, k_off=0
    terms.extend(strassen_2x2(2, 0, 0))
    # A22*B21: use Strassen with i_off=2, j_off=2, k_off=0
    terms.extend(strassen_2x2(2, 2, 0))

    # For C22 = A21*B12 + A22*B22:
    # A21*B12: use Strassen with i_off=2, j_off=0, k_off=2
    terms.extend(strassen_2x2(2, 0, 2))
    # A22*B22: use Strassen with i_off=2, j_off=2, k_off=2
    terms.extend(strassen_2x2(2, 2, 2))

    # That gives 8 * 7 = 56 terms for 4x4*4x4

    # For 4x4 * 4x3 (last 3 columns of B, k=4,5,6), use schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # k = 4, 5, 6
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
