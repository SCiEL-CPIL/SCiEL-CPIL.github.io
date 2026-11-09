import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 5

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, 1.0, atol=1e-8)

    def get_schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Build rank-65 decomposition: 49 Strassen terms for 4x4x4 + 16 schoolbook for k=4
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2 base
    def strassen_2x2():
        return [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]

    # For 4x4x4 Strassen, we have 7 outer terms, each with 7 inner terms
    # Outer term j determines which 2x2 blocks of A and B are combined
    # Inner term t determines the Strassen multiplication pattern

    # The 4x4x4 tensor: C[i,k] = sum_j A[i,j]*B[j,k] for i,k in 0..3
    # Split into 2x2 blocks: A = [[A00,A01],[A10,A11]] where A00 is top-left 2x2, etc.
    # Similarly for B and C.

    # Strassen 4x4x4: 49 multiplications
    # For each of 7 "outer" Strassen steps, we have 7 "inner" Strassen multiplications

    # Let me use a cleaner formulation. The 4x4 Strassen algorithm:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # But each of these is a 2x2x2 Strassen multiplication, which itself uses 7 multiplications.
    # So total = 7 * 7 = 49.

    # For the outer level, each "multiplication" like (A00+A11)*(B00+B11) is a 2x2x2 matrix multiplication.
    # This requires 7 scalar multiplications using Strassen.

    # Let me define the outer coefficients and inner coefficients properly.

    # Outer: 7 terms, each specifying linear combinations of 2x2 blocks
    # Inner: 7 terms, each specifying Strassen 2x2 pattern

    # For outer term j, we have:
    # U_outer[j] = linear combination of 2x2 blocks of A
    # V_outer[j] = linear combination of 2x2 blocks of B
    # The product contributes to C blocks according to W_outer[j]

    # Let me define this more carefully using the standard 4x4 Strassen.

    # Actually, let me just build it directly. For the 4x4x4 part (k=0..3):

    # The tensor T[i,j,k] for i,j,k in 0..3 (where i is row of A/C, j is col of A/row of B, k is col of B/C)
    # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k

    # Let me use the recursive Strassen construction.

    # Define 2x2 Strassen patterns
    s2x2 = [
        # (U_dict, V_dict) where dict maps (row,col) to coefficient
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
    ]

    # For 4x4, we split into 2x2 blocks. The outer Strassen has 7 terms.
    # Each outer term j has U_block and V_block which are 2x2 matrices of coefficients
    # for the 4 2x2 subblocks of A and B respectively.

    # Then for each inner Strassen term t, we combine to get the final U, V, W.

    # Let me define the outer 4x4 Strassen:
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)*B00 -> contributes to C10, C11
    # M3 = A00*(B01-B11) -> contributes to C01, C11
    # M4 = A11*(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)*B11 -> contributes to C00, C01
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00, C11

    # Where A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # Similarly for B and C.

    # For each outer term, the U factor is a 4x4 matrix (sparse) and V factor is a 4x4 matrix (sparse).
    # The W factor specifies how the product contributes to C.

    # Let me build this step by step.

    col = 0
    for j in range(7):
        # Define U_block[j] and V_block[j] as 2x2 arrays of 2x2 matrices
        # Each entry is a dict of (local_row, local_col) -> coefficient

        if j == 0:
            # M1 = (A00+A11)*(B00+B11)
            U_blk = [[{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}],
                     [{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}]]
            V_blk = [[{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}],
                     [{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}]]
            W_contrib = [((0,0),1), ((1,1),1)]  # C00 += M1, C11 += M1
        elif j == 1:
            # M2 = (A10+A11)*B00
            U_blk = [[{}, {}],
                     [{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}]]
            V_blk = [[{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {}],
                     [{}, {}]]
            W_contrib = [((1,0),1), ((1,1),-1)]  # C10 += M2, C11 -= M2
        elif j == 2:
            # M3 = A00*(B01-B11)
            U_blk = [[{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {}],
                     [{}, {}]]
            V_blk = [[{}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}],
                     [{}, {(0,0):-1,(0,1):-1,(1,0):-1,(1,1):-1}]]
            W_contrib = [((0,1),1), ((1,1),1)]  # C01 += M3, C11 += M3
        elif j == 3:
            # M4 = A11*(B10-B00)
            U_blk = [[{}, {}],
                     [{}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}]]
            V_blk = [[{(0,0):-1,(0,1):-1,(1,0):-1,(1,1):-1}, {}],
                     [{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {}]]
            W_contrib = [((0,0),1), ((1,0),1)]  # C00 += M4, C10 += M4
        elif j == 4:
            # M5 = (A00+A01)*B11
            U_blk = [[{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}],
                     [{}, {}]]
            V_blk = [[{}, {}],
                     [{}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}]]
            W_contrib = [((0,0),-1), ((0,1),1)]  # C00 -= M5, C01 += M5
        elif j == 5:
            # M6 = (A10-A00)*(B00+B01)
            U_blk = [[{(0,0):-1,(0,1):-1,(1,0):-1,(1,1):-1}, {}],
                     [{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {}]]
            V_blk = [[{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}],
                     [{}, {}]]
            W_contrib = [((1,1),1)]  # C11 += M6
        elif j == 6:
            # M7 = (A01-A11)*(B10+B11)
            U_blk = [[{}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}],
                     [{}, {(0,0):-1,(0,1):-1,(1,0):-1,(1,1):-1}]]
            V_blk = [[{}, {}],
                     [{(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}]]
            W_contrib = [((0,0),1), ((1,1),1)]  # C00 += M7, C11 += M7

        # Now for each of the 7 inner Strassen terms
        for t in range(7):
            u_inner, v_inner = s2x2[t]

            # Build U factor: 4x4 matrix
            U_4x4 = {}
            for br in range(2):
                for bc in range(2):
                    for (lr, lc), c in U_blk[br][bc].items():
                        for (ir, ic), ci in u_inner.items():
                            gr, gc = br*2+lr, bc*2+lc
                            U_4x4[(gr, gc)] = U_4x4.get((gr, gc), 0) + c * ci

            # Build V factor: 4x4 matrix
            V_4x4 = {}
            for br in range(2):
                for bc in range(2):
                    for (lr, lc), c in V_blk[br][bc].items():
                        for (ir, ic), ci in v_inner.items():
                            gr, gc = br*2+lr, bc*2+lc
                            V_4x4[(gr, gc)] = V_4x4.get((gr, gc), 0) + c * ci

            # Build W factor: contributes to C blocks
            W_4x4 = {}
            for (cbr, cbc), coeff in W_contrib:
                # The inner Strassen multiplication produces a 2x2 result
                # We need to figure out how it maps to C blocks
                # Actually, the W factor should be the identity for the C block
                # since we're just adding the product to the C block
                for (ir, ic) in [(0,0),(0,1),(1,0),(1,1)]:
                    gr, gc = cbr*2+ir, cbc*2+ic
                    W_4x4[(gr, gc)] = W_4x4.get((gr, gc), 0) + coeff

            # Fill in U, V, W arrays
            for (ai, aj), c in U_4x4.items():
                if c != 0:
                    U[ai * m + aj, col] += c
            for (bi, bj), c in V_4x4.items():
                if c != 0:
                    V[bi * p + bj, col] += c
            for (ci, cj), c in W_4x4.items():
                if c != 0:
                    W[cj * n + ci, col] += c
            col += 1

    # 16 schoolbook terms for column k=4
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1

    if time.time() > deadline - 5:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
