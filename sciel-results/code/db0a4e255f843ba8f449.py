import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6

    # Rank 42 solution using Strassen blocks.
    # 2x4 @ 4x6: split m=4 into two blocks of 2, p=6 into three blocks of 2.
    # For each (j_block, k_block) pair, use Strassen 2x2 @ 2x2 (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    rank = 42

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen 2x2 @ 2x2 decomposition:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # U indices: A[i,j] -> index i*m + j (local 2x2: i in {0,1}, j in {0,1})
    #   local index: i*2 + j
    # V indices: B[j,k] -> index j*p + k (local 2x2: j in {0,1}, k in {0,1})
    #   local index: j*2 + k
    # W indices: C[i,k] -> index k*n + i (local 2x2: i in {0,1}, k in {0,1})
    #   local index: k*2 + i

    # U entries for each Strassen product (local 2x2 A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    strassen_U = [
        {0: 1, 3: 1},       # P1: A00 + A11
        {2: 1, 3: 1},       # P2: A10 + A11
        {0: 1},             # P3: A00
        {3: 1},             # P4: A11
        {0: 1, 1: 1},       # P5: A00 + A01
        {2: 1, 0: -1},      # P6: A10 - A00
        {1: 1, 3: -1},      # P7: A01 - A11
    ]

    # V entries for each Strassen product (local 2x2 B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    strassen_V = [
        {0: 1, 3: 1},       # P1: B00 + B11
        {0: 1},             # P2: B00
        {1: 1, 3: -1},      # P3: B01 - B11
        {2: 1, 0: -1},      # P4: B10 - B00
        {3: 1},             # P5: B11
        {0: 1, 1: 1},       # P6: B00 + B01
        {2: 1, 3: 1},       # P7: B10 + B11
    ]

    # W entries for each Strassen product (local 2x2 C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    strassen_W = [
        {0: 1, 3: 1},       # P1: C00 + C11
        {2: 1, 3: -1},      # P2: C10 - C11
        {1: 1, 3: 1},       # P3: C01 + C11
        {0: 1, 2: 1},       # P4: C00 + C10
        {0: -1, 1: 1},      # P5: -C00 + C01
        {3: 1},             # P6: C11
        {0: 1},             # P7: C00
    ]

    col = 0
    for j_block in range(2):  # j = 0,1 and j = 2,3
        for k_block in range(3):  # k = 0,1 and k = 2,3 and k = 4,5
            for prod_idx in range(7):
                r = col + prod_idx

                # U: A[i, j] where i in {0,1}, j in {j_block*2, j_block*2+1}
                for local_idx, sign in strassen_U[prod_idx].items():
                    gi = local_idx // 2
                    gj = local_idx % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign

                # V: B[j, k] where j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                for local_idx, sign in strassen_V[prod_idx].items():
                    gj = local_idx // 2
                    gk = local_idx % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign

                # W: C[i, k] where i in {0,1}, k in {k_block*2, k_block*2+1}
                for local_idx, sign in strassen_W[prod_idx].items():
                    gi = local_idx // 2
                    gk = local_idx % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign

            col += 7

    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
