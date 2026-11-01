"""Matrix multiplication 2,2,16 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 16
    rank = n * m * p  # 64

    # Build the multiplication tensor T[i,j,k,c1,c2,c3]
    # where A[i,j] * B[j,k] contributes to C[i,k]
    # Tensor index: ((i*m + j), (j*p + k), (k*n + i))
    # We need: T[a, b, c] = 1 if there exist i,j,k such that
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # Otherwise T[a, b, c] = 0

    tensor = np.zeros((n*m, m*p, p*n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1

    # Try to find a lower-rank decomposition using integer/half-integer entries
    # Start with the schoolbook rank-64 decomposition and try to reduce

    # The Strassen-based approach for 2x2 matrices gives rank 7 per 2x2 block
    # For 2x2 * 2x16, we have 8 independent 2x2 blocks in the k dimension
    # But we can potentially share across blocks

    # Approach: Use the known rank-7 Strassen decomposition for each 2x2 block
    # This gives rank 7*8 = 56
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)

    # Strassen 2x2x2 decomposition:
    # P1 = A00*(B00+B11)
    # P2 = (A00+A10)*B01
    # P3 = A11*(B00-B10)
    # P4 = (A11-A01)*B11
    # P5 = (A01+A11)*(B00+B10)
    # P6 = (A00+A01)*B11
    # P7 = (A10-A11)*(B01+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P2 + P6
    # C10 = P3 + P5
    # C11 = P1 - P2 + P3 + P6

    strassen_U = [
        [1, 0, 0, 0, 0, 0, 0],   # A00
        [1, 1, 0, 0, 0, 0, 0],   # A10
        [0, 0, 1, -1, 1, 0, -1], # A11
        [0, 0, 0, 0, 1, 1, 0],   # A01
    ]
    strassen_V = [
        [1, 0, 1, 0, 1, 0, 0],   # B00
        [0, 1, 0, 0, 0, 1, 1],   # B01
        [1, 0, -1, 1, 1, 0, 1],  # B10
        [1, 0, 0, 1, 0, 1, 1],   # B11
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 0, 0, 1, 0],   # C01
        [0, 0, 1, 0, 1, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]

    # Verify Strassen decomposition
    U_s = np.array(strassen_U, dtype=float)  # (4, 7)
    V_s = np.array(strassen_V, dtype=float)  # (4, 7)
    W_s = np.array(strassen_W, dtype=float)  # (4, 7)

    # Check: sum_c U[a,c] * V[b,c] * W[c,c] should equal tensor[a,b,c]
    # For 2x2x2: a = i*2+j, b = j*2+k, c = k*2+i
    tensor_222 = np.zeros((4, 4, 4), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                tensor_222[i*2+j, j*2+k, k*2+i] = 1

    # Verify
    max_err = 0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                val = 0
                for cc in range(7):
                    val += U_s[a, cc] * V_s[b, cc] * W_s[c, cc]
                max_err = max(max_err, abs(val - tensor_222[a, b, c]))

    # Now build the full 2x2x16 decomposition using 8 Strassen blocks
    rank_total = 7 * 8  # 56
    U_full = np.zeros((n*m, rank_total), dtype=float)
    V_full = np.zeros((m*p, rank_total), dtype=float)
    W_full = np.zeros((p*n, rank_total), dtype=float)

    for block in range(8):
        # Block t handles columns 2t and 2t+1 of B and C
        col_start = 2 * block
        base = 7 * block

        for si in range(4):  # A row index
            U_full[si, base + si] = U_s[si, si]  # This is wrong, let me fix

    # Let me redo this properly
    U_full = np.zeros((n*m, rank_total), dtype=float)
    V_full = np.zeros((m*p, rank_total), dtype=float)
    W_full = np.zeros((p*n, rank_total), dtype=float)

    for block in range(8):
        col_start = 2 * block
        base = 7 * block

        for r in range(7):
            # U: A indices (i*m + j) = i*2 + j
            U_full[:, base + r] = 0
            V_full[:, base + r] = 0
            W_full[:, base + r] = 0

            # U: map strassen_U[r, :] to A indices
            # strassen_U[r, a_idx] where a_idx = i*2+j
            for a_idx in range(4):
                U_full[a_idx, base + r] = strassen_U[a_idx][r]

            # V: map strassen_V[r, b_idx] to B indices
            # strassen_V[r, b_idx] where b_idx = j*2+k
            # For block `block`, k = 2*block + (b_idx % 2), j = b_idx // 2
            for b_idx in range(4):
                j = b_idx // 2
                k_local = b_idx % 2
                k = 2 * block + k_local
                b_global = j * p + k
                V_full[b_global, base + r] = strassen_V[b_idx][r]

            # W: map strassen_W[r, c_idx] to C indices
            # strassen_W[r, c_idx] where c_idx = k*2+i
            # For block `block`, k = 2*block + (c_idx % 2), i = c_idx // 2
            for c_idx in range(4):
                i = c_idx // 2
                k_local = c_idx % 2
                k = 2 * block + k_local
                c_global = k * n + i
                W_full[c_global, base + r] = strassen_W[c_idx][r]

    # Verify the full decomposition
    max_err = 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = 0
                for r in range(rank_total):
                    val += U_full[a, r] * V_full[b, r] * W_full[c, r]
                max_err = max(max_err, abs(val - tensor[a, b, c]))

    # Check if we can do better with time
    elapsed = time.time() - start_time
    if elapsed < 100:
        # Try to find a lower-rank decomposition
        # The theoretical minimum rank for 2x2x16 is at least 2*2*16/sqrt(2) ~ 32
        # But Strassen gives 56. Let's see if we can do better.
        
        # One approach: use the fact that the 16 columns can be grouped
        # differently. For example, group into 4 blocks of 4 columns each.
        # Or use a different algebraic structure.
        
        # For now, return the rank-56 decomposition
        pass

    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    row_list.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-9:
                    row_list.append(int(round(val * 2)) / 2)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}
# EVOLVE-BLOCK-END
