"""Kronecker-embedded Strassen decomposition for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # We build each rank-1 slice as a list of (row, coeff) entries for U, V, W.
    # The 3x3 matrices are viewed as block matrices:
    #   A = [[A00, A01, A02],      B = [[B00, B01, B02],
    #        [A10, A11, A12],           [B10, B11, B12],
    #        [A20, A21, A22]]           [B20, B21, B22]]
    #
    # U index for A[i][j] is i*3 + j
    # V index for B[j][k] is j*3 + k
    # W index for C[i][k] is k*3 + i  (trace convention W[k*n+i])
    #
    # We decompose into 8 groups of rank-1 slices:
    #   Group 1: 2x2 Strassen on top-left blocks (7 slices)
    #   Group 2: a*b^T  (2x1 * 1x2) -> 4 slices
    #   Group 3: A21 * b  (2x2 * 2x1) -> 4 slices
    #   Group 4: a * b3  (2x1 * 1x1) -> 2 slices
    #   Group 5: a^T * B2 (1x2 * 2x2) -> 4 slices
    #   Group 6: a3 * b^T (1x1 * 1x2) -> 2 slices
    #   Group 7: a^T * b  (1x2 * 2x1) -> 2 slices
    #   Group 8: a3 * b3  (1x1 * 1x1) -> 1 slice
    # Total: 7+4+4+2+4+2+2+1 = 26

    def widx(k, i):
        """W index for C[i][k] using trace convention W[k*n+i]."""
        return k * n + i

    slices = []

    # --- Group 1: Strassen 2x2 on A[0:2,0:2] * B[0:2,0:2] ---
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=3, B11=4
    # C indices: C00=widx(0,0)=0, C01=widx(1,0)=3, C10=widx(0,1)=1, C11=widx(1,1)=4

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    slices.append(([ (0,1.0),(4,1.0) ], [ (0,1.0),(4,1.0) ], [ (widx(0,0),1.0),(widx(1,1),1.0) ]))
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    slices.append(([ (3,1.0),(4,1.0) ], [ (0,1.0) ], [ (widx(0,1),1.0),(widx(1,1),-1.0) ]))
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    slices.append(([ (0,1.0) ], [ (1,1.0),(4,-1.0) ], [ (widx(1,0),1.0),(widx(1,1),1.0) ]))
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    slices.append(([ (4,1.0) ], [ (3,1.0),(0,-1.0) ], [ (widx(0,0),1.0),(widx(0,1),1.0) ]))
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    slices.append(([ (0,1.0),(1,1.0) ], [ (4,1.0) ], [ (widx(0,0),-1.0),(widx(1,0),1.0) ]))
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    slices.append(([ (3,1.0),(0,-1.0) ], [ (0,1.0),(1,1.0) ], [ (widx(1,1),1.0) ]))
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    slices.append(([ (1,1.0),(4,-1.0) ], [ (3,1.0),(4,1.0) ], [ (widx(0,0),1.0) ]))

    # --- Group 2: a * b^T  where a=[A02,A12]^T, b^T=[B20,B21] ---
    # A indices: A02=2, A12=5
    # B indices: B20=6, B21=7
    # C: C00=widx(0,0), C01=widx(1,0), C10=widx(0,1), C11=widx(1,1)
    slices.append(([ (2,1.0) ], [ (6,1.0) ], [ (widx(0,0),1.0) ]))
    slices.append(([ (2,1.0) ], [ (7,1.0) ], [ (widx(1,0),1.0) ]))
    slices.append(([ (5,1.0) ], [ (6,1.0) ], [ (widx(0,1),1.0) ]))
    slices.append(([ (5,1.0) ], [ (7,1.0) ], [ (widx(1,1),1.0) ]))

    # --- Group 3: A21 * b  where A21 is A[0:2,0:2] (already in Strassen), b=[B02,B12]^T ---
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B02=2, B12=5
    # C: C02=widx(2,0)=6, C12=widx(2,1)=7
    slices.append(([ (0,1.0) ], [ (2,1.0) ], [ (widx(2,0),1.0) ]))
    slices.append(([ (1,1.0) ], [ (5,1.0) ], [ (widx(2,0),1.0) ]))
    slices.append(([ (3,1.0) ], [ (2,1.0) ], [ (widx(2,1),1.0) ]))
    slices.append(([ (4,1.0) ], [ (5,1.0) ], [ (widx(2,1),1.0) ]))

    # --- Group 4: a * b3  where a=[A02,A12]^T, b3=B22 ---
    # A indices: A02=2, A12=5
    # B index: B22=8
    # C: C02=widx(2,0)=6, C12=widx(2,1)=7
    slices.append(([ (2,1.0) ], [ (8,1.0) ], [ (widx(2,0),1.0) ]))
    slices.append(([ (5,1.0) ], [ (8,1.0) ], [ (widx(2,1),1.0) ]))

    # --- Group 5: a^T * B2  where a^T=[A20,A21], B2 is B[0:2,0:2] ---
    # A indices: A20=6, A21=7
    # B indices: B00=0, B01=1, B10=3, B11=4
    # C: C20=widx(0,2)=2, C21=widx(1,2)=5
    slices.append(([ (6,1.0) ], [ (0,1.0) ], [ (widx(0,2),1.0) ]))
    slices.append(([ (7,1.0) ], [ (3,1.0) ], [ (widx(0,2),1.0) ]))
    slices.append(([ (6,1.0) ], [ (1,1.0) ], [ (widx(1,2),1.0) ]))
    slices.append(([ (7,1.0) ], [ (4,1.0) ], [ (widx(1,2),1.0) ]))

    # --- Group 6: a3 * b^T  where a3=A22, b^T=[B20,B21] ---
    # A index: A22=8
    # B indices: B20=6, B21=7
    # C: C20=widx(0,2)=2, C21=widx(1,2)=5
    slices.append(([ (8,1.0) ], [ (6,1.0) ], [ (widx(0,2),1.0) ]))
    slices.append(([ (8,1.0) ], [ (7,1.0) ], [ (widx(1,2),1.0) ]))

    # --- Group 7: a^T * b  where a^T=[A20,A21], b=[B02,B12]^T ---
    # A indices: A20=6, A21=7
    # B indices: B02=2, B12=5
    # C: C22=widx(2,2)=8
    slices.append(([ (6,1.0) ], [ (2,1.0) ], [ (widx(2,2),1.0) ]))
    slices.append(([ (7,1.0) ], [ (5,1.0) ], [ (widx(2,2),1.0) ]))

    # --- Group 8: a3 * b3 ---
    # A index: A22=8, B index: B22=8
    # C: C22=widx(2,2)=8
    slices.append(([ (8,1.0) ], [ (8,1.0) ], [ (widx(2,2),1.0) ]))

    # Materialize the factor matrices from the slice definitions
    for r, (u_entries, v_entries, w_entries) in enumerate(slices):
        for idx, coeff in u_entries:
            U[idx][r] += coeff
        for idx, coeff in v_entries:
            V[idx][r] += coeff
        for idx, coeff in w_entries:
            W[idx][r] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
