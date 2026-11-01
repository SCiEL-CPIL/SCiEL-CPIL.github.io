import numpy as np
import time

def check_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition identity."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    r = U_arr.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U_arr[a * m + b] * V_arr[c * p + d] * W_arr[k * n + i])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 6

    # Overlapping Strassen decomposition for 2x3x6
    # A: 2x3, indices a*3+b (a=0,1; b=0,1,2)
    # B: 3x6, indices c*6+d (c=0,1,2; d=0,1,2,3,4,5)
    # C: 2x6, indices k*2+i (k=0..5; i=0,1)
    #
    # Strassen on A cols (0,1) for B cols (0,1,2) -> C cols (0,1,2)
    # Strassen on A cols (1,2) for B cols (3,4,5) -> C cols (3,4,5)
    # Cross terms for A col 1 * B cols (0..2) and A col 1 * B cols (3..5)
    #
    # Total: 7 + 6 + 7 + 6 + 6 = 32

    rank = 32
    U = np.zeros((6, rank))
    V = np.zeros((18, rank))
    W = np.zeros((12, rank))

    # ===== Block 1: Strassen on A(0,1) x B(0,1,2) -> C(0,1,2) =====
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B02=2, B10=6, B11=7, B12=8
    # C indices: C00=0, C10=1, C01=2, C11=3, C02=4, C12=5

    # P1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[7, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (A10+A11)B00 -> C10
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1

    # P3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[7, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # P4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[6, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # P5: (A00+A01)B11 -> C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[7, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # P6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (A01-A11)(B10+B11) -> C00, C11
    U[1, 6] = 1; U[4, 6] = -1
    V[6, 6] = 1; V[7, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1

    # Block 1 column 2 (B col 2): A(0,1) x B(0,1,2) -> C col 2
    # Q1: A00 * B02 -> C02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1

    # Q2: A01 * B12 -> C02
    U[1, 8] = 1
    V[8, 8] = 1
    W[4, 8] = 1

    # Q3: A10 * B02 -> C12
    U[3, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1

    # Q4: A11 * B12 -> C12
    U[4, 10] = 1
    V[8, 10] = 1
    W[5, 10] = 1

    # ===== Block 2: Strassen on A(1,2) x B(3,4,5) -> C(3,4,5) =====
    # A indices: A01=1, A02=2, A11=4, A12=5
    # B indices: B03=3, B04=4, B05=5, B13=9, B14=10, B15=11
    # C indices: C03=6, C13=7, C04=8, C14=9, C05=10, C15=11

    # P8: (A01+A12)(B03+B14) -> C03, C14
    U[1, 11] = 1; U[5, 11] = 1
    V[3, 11] = 1; V[10, 11] = 1
    W[6, 11] = 1; W[9, 11] = 1

    # P9: (A11+A12)B03 -> C13
    U[4, 12] = 1; U[5, 12] = 1
    V[3, 12] = 1
    W[7, 12] = 1

    # P10: A01(B04-B14) -> C04, C14
    U[1, 13] = 1
    V[4, 13] = 1; V[10, 13] = -1
    W[8, 13] = 1; W[9, 13] = 1

    # P11: A12(B13-B03) -> C03, C13
    U[5, 14] = 1
    V[9, 14] = 1; V[3, 14] = -1
    W[6, 14] = 1; W[7, 14] = 1

    # P12: (A01+A02)B14 -> C03, C04
    U[1, 15] = 1; U[2, 15] = 1
    V[10, 15] = 1
    W[6, 15] = -1; W[8, 15] = 1

    # P13: (A11-A01)(B03+B04) -> C14
    U[4, 16] = 1; U[1, 16] = -1
    V[3, 16] = 1; V[4, 16] = 1
    W[9, 16] = 1

    # P14: (A02-A12)(B13+B14) -> C03, C14
    U[2, 17] = 1; U[5, 17] = -1
    V[9, 17] = 1; V[10, 17] = 1
    W[6, 17] = 1; W[9, 17] = 1

    # Block 2 column 5 (B col 5): A(1,2) x B(3,4,5) -> C col 5
    # Q5: A01 * B05 -> C05
    U[1, 18] = 1
    V[5, 18] = 1
    W[10, 18] = 1

    # Q6: A02 * B15 -> C05
    U[2, 19] = 1
    V[11, 19] = 1
    W[10, 19] = 1

    # Q7: A11 * B05 -> C15
    U[4, 20] = 1
    V[5, 20] = 1
    W[11, 20] = 1

    # Q8: A12 * B15 -> C15
    U[5, 21] = 1
    V[11, 21] = 1
    W[11, 21] = 1

    # ===== Cross terms: A col 1 * B cols (0,1,2) and A col 1 * B cols (3,4,5) =====
    # These are needed because Strassen on A(0,1) uses A01, and Strassen on A(1,2) uses A01,
    # but the cross terms A01*B0j and A01*B1j for j in the overlap need to be accounted for.
    # Actually, in the block decomposition, A01 appears in both Strassen blocks, so the
    # contributions from A01*B(0,1,2) and A01*B(3,4,5) are already included in the Strassen
    # factors. The cross terms are for A02*B(0,1,2) and A01*B(3,4,5) which are NOT in
    # the Strassen blocks.
    #
    # Wait, let me reconsider. The Strassen on A(0,1) covers A00, A01, A10, A11.
    # The Strassen on A(1,2) covers A01, A02, A11, A12.
    # A01 and A11 are shared. The contributions from A01 and A11 to C(0,1,2) come from
    # Block 1's Strassen. The contributions from A01 and A11 to C(3,4,5) come from
    # Block 2's Strassen.
    # What's missing? A02*B(0,1,2) -> C(0,1,2) and A01*B(3,4,5) is in Block 2.
    # Actually A02*B(0,1,2) -> C(0,1,2) is NOT covered. We need 6 rank-1 terms for this.
    # Similarly, A01*B(3,4,5) -> C(3,4,5) IS covered by Block 2's Strassen.
    #
    # Let me re-examine: Block 1 Strassen covers A00,A01,A10,A11 times B(0,1,2) -> C(0,1,2).
    # Block 2 Strassen covers A01,A02,A11,A12 times B(3,4,5) -> C(3,4,5).
    # Missing: A02*B(0,1,2) -> C(0,1,2) and A00,A10 times B(3,4,5) -> C(3,4,5).
    # That's 6 + 6 = 12 more terms. Total would be 32+12 = 44. That's worse.
    #
    # I think the overlapping approach doesn't save rank in this case because the
    # shared columns (A01, A11) don't reduce the total count. Let me just verify
    # the current 32-term construction and fall back if it fails.

    if time.time() > deadline:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    if check_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: use the known rank-34 block decomposition
    # 2x3x3 rank 17, doubled for 2x3x6
    rank34 = 34
    U34 = np.zeros((6, rank34))
    V34 = np.zeros((18, rank34))
    W34 = np.zeros((12, rank34))

    # 2x3x3 decomposition (rank 17)
    # A: 2x3, indices a*3+b
    # B: 3x3, indices c*3+d
    # C: 2x3, indices k*2+i

    # Strassen on A(0,1) x B(0,1) -> C(0,1)
    # P1: (A00+A11)(B00+B11)
    U34[0, 0] = 1; U34[4, 0] = 1
    V34[0, 0] = 1; V34[4, 0] = 1
    W34[0, 0] = 1; W34[3, 0] = 1

    # P2: (A10+A11)B00
    U34[3, 1] = 1; U34[4, 1] = 1
    V34[0, 1] = 1
    W34[1, 1] = 1

    # P3: A00(B01-B11)
    U34[0, 2] = 1
    V34[1, 2] = 1; V34[4, 2] = -1
    W34[2, 2] = 1; W34[3, 2] = 1

    # P4: A11(B10-B00)
    U34[4, 3] = 1
    V34[3, 3] = 1; V34[0, 3] = -1
    W34[0, 3] = 1; W34[1, 3] = 1

    # P5: (A00+A01)B11
    U34[0, 4] = 1; U34[1, 4] = 1
    V34[4, 4] = 1
    W34[0, 4] = -1; W34[2, 4] = 1

    # P6: (A10-A00)(B00+B01)
    U34[3, 5] = 1; U34[0, 5] = -1
    V34[0, 5] = 1; V34[1, 5] = 1
    W34[3, 5] = 1

    # P7: (A01-A11)(B10+B11)
    U34[1, 6] = 1; U34[4, 6] = -1
    V34[3, 6] = 1; V34[4, 6] = 1
    W34[0, 6] = 1; W34[3, 6] = 1

    # Column 2 of B: A(0,1,2) x B(0,1,2) col 2 -> C col 2
    # Q1: A00*B02
    U34[0, 7] = 1
    V34[2, 7] = 1
    W34[4, 7] = 1

    # Q2: A01*B12
    U34[1, 8] = 1
    V34[5, 8] = 1
    W34[4, 8] = 1

    # Q3: A10*B02
    U34[3, 9] = 1
    V34[2, 9] = 1
    W34[5, 9] = 1

    # Q4: A11*B12
    U34[4, 10] = 1
    V34[5, 10] = 1
    W34[5, 10] = 1

    # A col 2 x B(0,1,2) -> C(0,1,2)
    # R1: A02*B20
    U34[2, 11] = 1
    V34[6, 11] = 1
    W34[0, 11] = 1

    # R2: A02*B21
    U34[2, 12] = 1
    V34[7, 12] = 1
    W34[2, 12] = 1

    # R3: A02*B22
    U34[2, 13] = 1
    V34[8, 13] = 1
    W34[4, 13] = 1

    # R4: A12*B20
    U34[5, 14] = 1
    V34[6, 14] = 1
    W34[1, 14] = 1

    # R5: A12*B21
    U34[5, 15] = 1
    V34[7, 15] = 1
    W34[3, 15] = 1

    # R6: A12*B22
    U34[5, 16] = 1
    V34[8, 16] = 1
    W34[5, 16] = 1

    # Verify 2x3x3
    if check_decomposition(U34[:, :17], V34[:, :17], W34[:, :17], 2, 3, 3):
        # Combine two 2x3x3 blocks
        U_comb = np.zeros((6, 34))
        U_comb[:, :17] = U34[:, :17]
        U_comb[:, 17:] = U34[:, :17]

        V_comb = np.zeros((18, 34))
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V34[j * 3 + k, :17]
                V_comb[j * 6 + 3 + k, 17:] = V34[j * 3 + k, :17]

        W_comb = np.zeros((12, 34))
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W34[k * 2 + i, :17]
                W_comb[(k + 3) * 2 + i, 17:] = W34[k * 2 + i, :17]

        if check_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
            return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}

    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
