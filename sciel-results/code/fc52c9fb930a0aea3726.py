"""Matrix multiplication 2,4,6 — rank-42 decomposition via Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)

    def add_strassen_block(i_start, j_start, k_start, offset):
        # A: rows i_start..i_start+1, cols j_start..j_start+1
        # B: rows j_start..j_start+1, cols k_start..k_start+1
        # C: rows k_start..k_start+1, cols i_start..i_start+1
        # U index: (i+m_col) where i is row of A, m_col is col of A
        # V index: (j_row + k_col) where j_row is row of B, k_col is col of B
        # W index: (k_row * n + i_col) where k_row is row of C, i_col is col of C

        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        c = offset
        U[(i_start+0)*m + (j_start+0), c] += 1
        U[(i_start+1)*m + (j_start+1), c] += 1
        V[(j_start+0)*p + (k_start+0), c] += 1
        V[(j_start+1)*p + (k_start+1), c] += 1
        W[(k_start+0)*n + (i_start+0), c] += 1
        W[(k_start+1)*n + (i_start+1), c] += 1

        # M2: (A10+A11)B00 -> C10+=1, C11-=1
        c = offset + 1
        U[(i_start+1)*m + (j_start+0), c] += 1
        U[(i_start+1)*m + (j_start+1), c] += 1
        V[(j_start+0)*p + (k_start+0), c] += 1
        W[(k_start+0)*n + (i_start+1), c] += 1
        W[(k_start+1)*n + (i_start+1), c] -= 1

        # M3: A00(B01-B11) -> C01+=1, C11+=1
        c = offset + 2
        U[(i_start+0)*m + (j_start+0), c] += 1
        V[(j_start+0)*p + (k_start+1), c] += 1
        V[(j_start+1)*p + (k_start+1), c] -= 1
        W[(k_start+1)*n + (i_start+0), c] += 1
        W[(k_start+1)*n + (i_start+1), c] += 1

        # M4: A11(B10-B00) -> C00+=1, C10+=1
        c = offset + 3
        U[(i_start+1)*m + (j_start+1), c] += 1
        V[(j_start+1)*p + (k_start+0), c] += 1
        V[(j_start+0)*p + (k_start+0), c] -= 1
        W[(k_start+0)*n + (i_start+0), c] += 1
        W[(k_start+0)*n + (i_start+1), c] += 1

        # M5: (A00+A01)B11 -> C00-=1, C01+=1
        c = offset + 4
        U[(i_start+0)*m + (j_start+0), c] += 1
        U[(i_start+0)*m + (j_start+1), c] += 1
        V[(j_start+1)*p + (k_start+1), c] += 1
        W[(k_start+0)*n + (i_start+0), c] -= 1
        W[(k_start+1)*n + (i_start+0), c] += 1

        # M6: (A10-A00)(B00+B01) -> C11+=1
        c = offset + 5
        U[(i_start+1)*m + (j_start+0), c] += 1
        U[(i_start+0)*m + (j_start+0), c] -= 1
        V[(j_start+0)*p + (k_start+0), c] += 1
        V[(j_start+0)*p + (k_start+1), c] += 1
        W[(k_start+1)*n + (i_start+1), c] += 1

        # M7: (A01-A11)(B10+B11) -> C00+=1
        c = offset + 6
        U[(i_start+0)*m + (j_start+1), c] += 1
        U[(i_start+1)*m + (j_start+1), c] -= 1
        V[(j_start+1)*p + (k_start+0), c] += 1
        V[(j_start+1)*p + (k_start+1), c] += 1
        W[(k_start+0)*n + (i_start+0), c] += 1

    # 6 Strassen blocks: 2 A-column-pairs x 3 B-column-pairs
    for a_col in [0, 2]:
        for b_col in [0, 2, 4]:
            offset = (a_col // 2) * 21 + (b_col // 2) * 7
            add_strassen_block(0, a_col, b_col, offset)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
