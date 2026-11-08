"""Kronecker-embedded Strassen decomposition for 3x3 matrix multiplication.

Treats 3x3 matrices as 2x2 block matrices with 1x1 off-diagonal blocks.
Uses rank-7 Strassen for the 2x2 block product, and shares intermediate
linear forms across the block interactions to minimize total rank.

Block structure:
  A = [[A00, A01, a],     B = [[B00, B01, b],
       [A10, A11, c],          [B10, B11, d],
       [e,   f,   g]]          [h,   i,   k]]

where:
  A00 = A[0,0], A01 = A[0,1], a = A[0,2]
  A10 = A[1,0], A11 = A[1,1], c = A[1,2]
  e = A[2,0],   f = A[2,1],   g = A[2,2]

  B00 = B[0,0], B01 = B[0,1], b = B[0,2]
  B10 = B[1,0], B11 = B[1,1], d = B[1,2]
  h = B[2,0],   i = B[2,1],   k = B[2,2]

C = A*B:
  C00 = A00*B00 + A01*B10 + a*h
  C01 = A00*B01 + A01*B11 + a*i
  C02 = A00*b + A01*d + a*k
  C10 = A10*B00 + A11*B10 + c*h
  C11 = A10*B01 + A11*B11 + c*i
  C12 = A10*b + A11*d + c*k
  C20 = e*B00 + f*B10 + g*h
  C21 = e*B01 + f*B11 + g*i
  C22 = e*b + f*d + g*k

W index convention: W[k*n+i] for C[i,k], so W index = k*3 + i
"""

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # U index for A[i][j] is i*3 + j
    # V index for B[j][k] is j*3 + k
    # W index for C[i][k] is k*3 + i  (trace convention)

    def widx(k, i):
        return k * n + i

    slices = []

    # === Group 1: Strassen 2x2 on A[0:2,0:2] * B[0:2,0:2] (7 slices) ===
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=3, B11=4
    # C00=widx(0,0)=0, C01=widx(1,0)=3, C10=widx(0,1)=1, C11=widx(1,1)=4

    # M1 = (A00+A11)(B00+B11)
    slices.append(([ (0,1.0),(4,1.0) ], [ (0,1.0),(4,1.0) ], [ (widx(0,0),1.0),(widx(1,1),1.0) ]))
    # M2 = (A10+A11)B00
    slices.append(([ (3,1.0),(4,1.0) ], [ (0,1.0) ], [ (widx(0,1),1.0),(widx(1,1),-1.0) ]))
    # M3 = A00(B01-B11)
    slices.append(([ (0,1.0) ], [ (1,1.0),(4,-1.0) ], [ (widx(1,0),1.0),(widx(1,1),1.0) ]))
    # M4 = A11(B10-B00)
    slices.append(([ (4,1.0) ], [ (3,1.0),(0,-1.0) ], [ (widx(0,0),1.0),(widx(0,1),1.0) ]))
    # M5 = (A00+A01)B11
    slices.append(([ (0,1.0),(1,1.0) ], [ (4,1.0) ], [ (widx(0,0),-1.0),(widx(1,0),1.0) ]))
    # M6 = (A10-A00)(B00+B01)
    slices.append(([ (3,1.0),(0,-1.0) ], [ (0,1.0),(1,1.0) ], [ (widx(1,1),1.0) ]))
    # M7 = (A01-A11)(B10+B11)
    slices.append(([ (1,1.0),(4,-1.0) ], [ (3,1.0),(4,1.0) ], [ (widx(0,0),1.0) ]))

    # === Group 2: a * b^T where a=[A02,A12]^T, b^T=[B20,B21] (4 slices) ===
    # A02=2, A12=5; B20=6, B21=7
    slices.append(([ (2,1.0) ], [ (6,1.0) ], [ (widx(0,0),1.0) ]))
    slices.append(([ (2,1.0) ], [ (7,1.0) ], [ (widx(1,0),1.0) ]))
    slices.append(([ (5,1.0) ], [ (6,1.0) ], [ (widx(0,1),1.0) ]))
    slices.append(([ (5,1.0) ], [ (7,1.0) ], [ (widx(1,1),1.0) ]))

    # === Group 3: A[0:2,0:2] * b_col where b_col=[B02,B12]^T (4 slices) ===
    # A00=0, A01=1, A10=3, A11=4; B02=2, B12=5
    # C02=widx(2,0)=6, C12=widx(2,1)=7
    slices.append(([ (0,1.0) ], [ (2,1.0) ], [ (widx(2,0),1.0) ]))
    slices.append(([ (1,1.0) ], [ (5,1.0) ], [ (widx(2,0),1.0) ]))
    slices.append(([ (3,1.0) ], [ (2,1.0) ], [ (widx(2,1),1.0) ]))
    slices.append(([ (4,1.0) ], [ (5,1.0) ], [ (widx(2,1),1.0) ]))

    # === Group 4: a * k where a=[A02,A12]^T, k=B22 (2 slices) ===
    # A02=2, A12=5; B22=8
    slices.append(([ (2,1.0) ], [ (8,1.0) ], [ (widx(2,0),1.0) ]))
    slices.append(([ (5,1.0) ], [ (8,1.0) ], [ (widx(2,1),1.0) ]))

    # === Group 5: a_row^T * B[0:2,0:2] where a_row^T=[A20,A21] (4 slices) ===
    # A20=6, A21=7; B00=0, B01=1, B10=3, B11=4
    # C20=widx(0,2)=2, C21=widx(1,2)=5
    slices.append(([ (6,1.0) ], [ (0,1.0) ], [ (widx(0,2),1.0) ]))
    slices.append(([ (7,1.0) ], [ (3,1.0) ], [ (widx(0,2),1.0) ]))
    slices.append(([ (6,1.0) ], [ (1,1.0) ], [ (widx(1,2),1.0) ]))
    slices.append(([ (7,1.0) ], [ (4,1.0) ], [ (widx(1,2),1.0) ]))

    # === Group 6: g * b^T where g=A22, b^T=[B20,B21] (2 slices) ===
    # A22=8; B20=6, B21=7
    slices.append(([ (8,1.0) ], [ (6,1.0) ], [ (widx(0,2),1.0) ]))
    slices.append(([ (8,1.0) ], [ (7,1.0) ], [ (widx(1,2),1.0) ]))

    # === Group 7: a_row^T * b_col where a_row^T=[A20,A21], b_col=[B02,B12]^T (2 slices) ===
    # A20=6, A21=7; B02=2, B12=5
    # C22=widx(2,2)=8
    slices.append(([ (6,1.0) ], [ (2,1.0) ], [ (widx(2,2),1.0) ]))
    slices.append(([ (7,1.0) ], [ (5,1.0) ], [ (widx(2,2),1.0) ]))

    # === Group 8: g * k (1 slice) ===
    # A22=8, B22=8
    slices.append(([ (8,1.0) ], [ (8,1.0) ], [ (widx(2,2),1.0) ]))

    # Total: 7+4+4+2+4+2+2+1 = 26
    assert len(slices) == rank, f"Expected {rank} slices, got {len(slices)}"

    # Materialize the factor matrices from the slice definitions
    for r, (u_entries, v_entries, w_entries) in enumerate(slices):
        for idx, coeff in u_entries:
            U[idx][r] += coeff
        for idx, coeff in v_entries:
            V[idx][r] += coeff
        for idx, coeff in w_entries:
            W[idx][r] += coeff

    return {"U": U, "V": V, "W": W}
