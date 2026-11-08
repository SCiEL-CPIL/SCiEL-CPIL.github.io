"""3x3 matrix multiplication tensor decomposition via Kronecker-embedded Strassen."""
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def widx(k, i):
        return k * n + i

    # --- 7 Strassen slices on the top-left 2x2 block ---
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=3, B11=4
    # C: C00=widx(0,0)=0, C01=widx(1,0)=3, C10=widx(0,1)=1, C11=widx(1,1)=4

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] += 1.0; U[4, 0] += 1.0
    V[0, 0] += 1.0; V[4, 0] += 1.0
    W[widx(0,0), 0] += 1.0; W[widx(1,1), 0] += 1.0

    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[3, 1] += 1.0; U[4, 1] += 1.0
    V[0, 1] += 1.0
    W[widx(0,1), 1] += 1.0; W[widx(1,1), 1] += -1.0

    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] += 1.0
    V[1, 2] += 1.0; V[4, 2] += -1.0
    W[widx(1,0), 2] += 1.0; W[widx(1,1), 2] += 1.0

    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[4, 3] += 1.0
    V[3, 3] += 1.0; V[0, 3] += -1.0
    W[widx(0,0), 3] += 1.0; W[widx(0,1), 3] += 1.0

    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] += 1.0; U[1, 4] += 1.0
    V[4, 4] += 1.0
    W[widx(0,0), 4] += -1.0; W[widx(1,0), 4] += 1.0

    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[3, 5] += 1.0; U[0, 5] += -1.0
    V[0, 5] += 1.0; V[1, 5] += 1.0
    W[widx(1,1), 5] += 1.0

    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] += 1.0; U[4, 6] += -1.0
    V[3, 6] += 1.0; V[4, 6] += 1.0
    W[widx(0,0), 6] += 1.0

    # --- Off-diagonal block interactions (16 slices) ---
    # We need to compute:
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22

    # Slice 7: A02*B20 -> C00
    U[2, 7] += 1.0; V[6, 7] += 1.0; W[widx(0,0), 7] += 1.0
    # Slice 8: A12*B21 -> C11
    U[5, 8] += 1.0; V[7, 8] += 1.0; W[widx(1,1), 8] += 1.0
    # Slice 9: A20*B20 -> C20
    U[6, 9] += 1.0; V[6, 9] += 1.0; W[widx(0,2), 9] += 1.0
    # Slice 10: A21*B21 -> C21
    U[7, 10] += 1.0; V[7, 10] += 1.0; W[widx(1,2), 10] += 1.0
    # Slice 11: A22*B22 -> C22
    U[8, 11] += 1.0; V[8, 11] += 1.0; W[widx(2,2), 11] += 1.0

    # Slice 12: A00*B02 -> C02
    U[0, 12] += 1.0; V[2, 12] += 1.0; W[widx(2,0), 12] += 1.0
    # Slice 13: A01*B12 -> C02
    U[1, 13] += 1.0; V[5, 13] += 1.0; W[widx(2,0), 13] += 1.0
    # Slice 14: A10*B02 -> C12
    U[3, 14] += 1.0; V[2, 14] += 1.0; W[widx(2,1), 14] += 1.0
    # Slice 15: A11*B12 -> C12
    U[4, 15] += 1.0; V[5, 15] += 1.0; W[widx(2,1), 15] += 1.0

    # Slice 16: A20*B00 -> C20
    U[6, 16] += 1.0; V[0, 16] += 1.0; W[widx(0,2), 16] += 1.0
    # Slice 17: A21*B10 -> C20
    U[7, 17] += 1.0; V[3, 17] += 1.0; W[widx(0,2), 17] += 1.0
    # Slice 18: A20*B01 -> C21
    U[6, 18] += 1.0; V[1, 18] += 1.0; W[widx(1,2), 18] += 1.0
    # Slice 19: A21*B11 -> C21
    U[7, 19] += 1.0; V[4, 19] += 1.0; W[widx(1,2), 19] += 1.0

    # Slice 20: A02*B22 -> C02
    U[2, 20] += 1.0; V[8, 20] += 1.0; W[widx(2,0), 20] += 1.0
    # Slice 21: A12*B22 -> C12
    U[5, 21] += 1.0; V[8, 21] += 1.0; W[widx(2,1), 21] += 1.0
    # Slice 22: A22*B20 -> C20
    U[8, 22] += 1.0; V[6, 22] += 1.0; W[widx(0,2), 22] += 1.0
    # Slice 23: A22*B21 -> C21
    U[8, 23] += 1.0; V[7, 23] += 1.0; W[widx(1,2), 23] += 1.0
    # Slice 24: A20*B02 -> C22
    U[6, 24] += 1.0; V[2, 24] += 1.0; W[widx(2,2), 24] += 1.0
    # Slice 25: A21*B12 -> C22
    U[7, 25] += 1.0; V[5, 25] += 1.0; W[widx(2,2), 25] += 1.0

    # Wait, that's 26 slices. Let me recount.
    # Actually I need to be more careful. Let me list all 27 terms of 3x3 matrix mult:
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22

    # The Strassen 7 slices handle C00, C01, C10, C11 (the top-left 2x2 block).
    # Remaining terms:
    # C02: A00*B02, A01*B12, A02*B22 (3 terms)
    # C12: A10*B02, A11*B12, A12*B22 (3 terms)
    # C20: A20*B00, A21*B10, A22*B20 (3 terms)
    # C21: A20*B01, A21*B11, A22*B21 (3 terms)
    # C22: A20*B02, A21*B12, A22*B22 (3 terms)
    # Total: 15 terms

    # Plus the off-diagonal terms that contribute to C00, C01, C10, C11:
    # C00 also has A02*B20
    # C01 also has A02*B21
    # C10 also has A12*B20
    # C11 also has A12*B21
    # That's 4 more terms.

    # Total: 19 terms + 7 Strassen = 26? No wait, Strassen replaces 4 terms with 7.
    # So we have 9*3 = 27 terms total. Strassen handles 4 of them in 7 slices.
    # Remaining: 27 - 4 = 23 terms, but some can be shared.

    # Let me just use rank 26 as in the parent.

    # Reset and redo with rank 26
    rank = 26
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen slices 0-6 (as above)
    U[0, 0] += 1.0; U[4, 0] += 1.0; V[0, 0] += 1.0; V[4, 0] += 1.0
    W[0, 0] += 1.0; W[4, 0] += 1.0

    U[3, 1] += 1.0; U[4, 1] += 1.0; V[0, 1] += 1.0
    W[1, 1] += 1.0; W[4, 1] += -1.0

    U[0, 2] += 1.0; V[1, 2] += 1.0; V[4, 2] += -1.0
    W[3, 2] += 1.0; W[4, 2] += 1.0

    U[4, 3] += 1.0; V[3, 3] += 1.0; V[0, 3] += -1.0
    W[0, 3] += 1.0; W[1, 3] += 1.0

    U[0, 4] += 1.0; U[1, 4] += 1.0; V[4, 4] += 1.0
    W[0, 4] += -1.0; W[3, 4] += 1.0

    U[3, 5] += 1.0; U[0, 5] += -1.0; V[0, 5] += 1.0; V[1, 5] += 1.0
    W[4, 5] += 1.0

    U[1, 6] += 1.0; U[4, 6] += -1.0; V[3, 6] += 1.0; V[4, 6] += 1.0
    W[0, 6] += 1.0

    # 19 remaining direct multiplication terms
    terms = [
        # (A_idx, B_idx, C_i, C_k)
        (2, 6, 0, 0),   # A02*B20 -> C00
        (2, 7, 0, 1),   # A02*B21 -> C01
        (5, 6, 1, 0),   # A12*B20 -> C10
        (5, 7, 1, 1),   # A12*B21 -> C11
        (0, 2, 0, 2),   # A00*B02 -> C02
        (1, 5, 0, 2),   # A01*B12 -> C02
        (2, 8, 0, 2),   # A02*B22 -> C02
        (3, 2, 1, 2),   # A10*B02 -> C12
        (4, 5, 1, 2),   # A11*B12 -> C12
        (5, 8, 1, 2),   # A12*B22 -> C12
        (6, 0, 2, 0),   # A20*B00 -> C20
        (7, 3, 2, 0),   # A21*B10 -> C20
        (8, 6, 2, 0),   # A22*B20 -> C20
        (6, 1, 2, 1),   # A20*B01 -> C21
        (7, 4, 2, 1),   # A21*B11 -> C21
        (8, 7, 2, 1),   # A22*B21 -> C21
        (6, 2, 2, 2),   # A20*B02 -> C22
        (7, 5, 2, 2),   # A21*B12 -> C22
        (8, 8, 2, 2),   # A22*B22 -> C22
    ]

    for i, (ai, bi, ci, ck) in enumerate(terms):
        r = 7 + i
        U[ai, r] += 1.0
        V[bi, r] += 1.0
        W[widx(ck, ci), r] += 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
