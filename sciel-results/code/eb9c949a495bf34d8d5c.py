"""Direct rank-44 decomposition for the 3x4 x 4x4 matrix multiplication tensor.

Uses Strassen (rank 7) on four 2x2x2 sub-blocks where A has 2 rows and B has 2x2,
and schoolbook (rank 4) on four 1x2x2 sub-blocks. Total rank = 4*7 + 4*4 = 44.
"""

import time

N, M, P = 3, 4, 4
TIME_LIMIT = 120.0
SAFETY_MARGIN = 20.0


def _build_rank44():
    """Build rank-44 decomposition using Strassen on 2x2x2 blocks and schoolbook on 1x2x2 blocks."""
    width = 44
    U = [[0.0] * width for _ in range(N * M)]
    V = [[0.0] * width for _ in range(M * P)]
    W = [[0.0] * width for _ in range(P * N)]
    next_col = [0]

    def strassen_block(i0, r0, k0):
        """Apply Strassen to 2x2 A-block (rows i0:i0+2, cols r0:r0+2) * 2x2 B-block (rows r0:r0+2, cols k0:k0+2)."""
        iR0, iR1 = i0, i0 + 1
        aR0, aR1 = r0, r0 + 1
        kC0, kC1 = k0, k0 + 1

        # A-forms: list of (i, r, coeff) for each M-term
        AF = {
            1: [(iR0, aR0, 1), (iR1, aR1, 1)],       # A00 + A11
            2: [(iR1, aR0, 1), (iR1, aR1, 1)],       # A10 + A11
            3: [(iR0, aR0, 1)],                       # A00
            4: [(iR1, aR1, 1)],                       # A11
            5: [(iR0, aR0, 1), (iR0, aR1, 1)],       # A00 + A01
            6: [(iR1, aR0, 1), (iR0, aR0, -1)],      # A10 - A00
            7: [(iR0, aR1, 1), (iR1, aR1, -1)],      # A01 - A11
        }
        # B-forms: list of (r, k, coeff) for each M-term
        BF = {
            1: [(aR0, kC0, 1), (aR1, kC1, 1)],       # B00 + B11
            2: [(aR0, kC0, 1)],                       # B00
            3: [(aR0, kC1, 1), (aR1, kC1, -1)],      # B01 - B11
            4: [(aR0, kC0, 1), (aR1, kC0, 1)],       # B00 + B10
            5: [(aR1, kC1, 1)],                       # B11
            6: [(aR0, kC0, 1), (aR0, kC1, 1)],       # B00 + B01
            7: [(aR1, kC0, 1), (aR1, kC1, 1)],       # B10 + B11
        }
        # C-forms: list of (i, k, coeff) for each M-term
        CF = {
            1: [(iR0, kC0, 1), (iR1, kC1, 1)],       # C00 + C11
            2: [(iR1, kC0, 1), (iR1, kC1, -1)],      # C10 - C11
            3: [(iR0, kC1, 1), (iR1, kC1, 1)],       # C01 + C11
            4: [(iR0, kC0, 1), (iR1, kC0, 1)],       # C00 + C10
            5: [(iR0, kC0, -1), (iR0, kC1, 1)],      # -C00 + C01
            6: [(iR1, kC1, 1)],                       # C11
            7: [(iR0, kC0, 1)],                       # C00
        }

        for m in range(1, 8):
            if next_col[0] >= width:
                raise RuntimeError("exceeded width in strassen_block")
            c = next_col[0]
            next_col[0] += 1
            for (i, r, sa) in AF[m]:
                U[i * M + r][c] += sa
            for (r, k, sb) in BF[m]:
                V[r * P + k][c] += sb
            for (i, k, sc) in CF[m]:
                W[k * N + i][c] += sc

    def naive_block(i_rows, r_rows, k_cols):
        """Schoolbook for block: one column per (i, r, k) triple."""
        for i in i_rows:
            for r in r_rows:
                for k in k_cols:
                    if next_col[0] >= width:
                        raise RuntimeError("exceeded width in naive_block")
                    c = next_col[0]
                    next_col[0] += 1
                    U[i * M + r][c] = 1.0
                    V[r * P + k][c] = 1.0
                    W[k * N + i][c] = 1.0

    # Four 2x2x2 Strassen blocks (A rows {0,1} x B rows {r0,r0+1} x B cols {k0,k0+1})
    strassen_block(0, 0, 0)  # rows{0,1} x Brows{0,1} x Bcols{0,1}
    strassen_block(0, 2, 0)  # rows{0,1} x Brows{2,3} x Bcols{0,1}
    strassen_block(0, 0, 2)  # rows{0,1} x Brows{0,1} x Bcols{2,3}
    strassen_block(0, 2, 2)  # rows{0,1} x Brows{2,3} x Bcols{2,3}

    # Four 1x2x2 naive blocks (A row {2} x B rows {r0,r0+1} x B cols {k0,k0+1})
    naive_block([2], [0, 1], [0, 1])  # row{2} x Brows{0,1} x Bcols{0,1}
    naive_block([2], [2, 3], [0, 1])  # row{2} x Brows{2,3} x Bcols{0,1}
    naive_block([2], [0, 1], [2, 3])  # row{2} x Brows{0,1} x Bcols{2,3}
    naive_block([2], [2, 3], [2, 3])  # row{2} x Brows{2,3} x Bcols{2,3}

    rank = next_col[0]
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W, "rank": rank}


def _schoolbook():
    """Guaranteed-correct rank-48 schoolbook decomposition."""
    rank = N * M * P
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for r in range(M):
            for k in range(P):
                U[i * M + r][col] = 1.0
                V[r * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(res):
    """Verify that sum_c U[i*M+r][c] * V[r*P+k][c] * W[k*N+i][c] == 1 for all i,r,k."""
    U, V, W = res["U"], res["V"], res["W"]
    rank = len(U[0]) if U and U[0] else 0
    for i in range(N):
        for r in range(M):
            for k in range(P):
                s = 0.0
                for c in range(rank):
                    s += U[i * M + r][c] * V[r * P + k][c] * W[k * N + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    start = time.time()
    deadline = TIME_LIMIT - SAFETY_MARGIN
    try:
        res = _build_rank44()
        if time.time() < deadline and _verify(res):
            return {"U": res["U"], "V": res["V"], "W": res["W"]}
    except Exception:
        pass
    # Fallback: guaranteed-correct schoolbook
    sb = _schoolbook()
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
