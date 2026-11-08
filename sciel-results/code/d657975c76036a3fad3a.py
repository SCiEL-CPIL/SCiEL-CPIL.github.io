"""2x4 by 4x6 matrix multiplication, exact rank-42 tensor decomposition.

Uses block Strassen: A (2x4) split into two 2x2 column blocks, B (4x6) split
into six 2x2 blocks (2 row-blocks x 3 col-blocks). Each 2x2 x 2x2 multiply
uses the 7-term Strassen scheme. Total rank = 6 * 7 = 42.
"""

import time as _time

_DEADLINE = 130.0  # safety margin before 150s kill


def construct():
    _t0 = _time.monotonic()

    n, m, p = 2, 4, 6
    rank = 42

    U = [[0] * rank for _ in range(n * m)]   # 8  x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # Strassen 7-term coefficient tables for 2x2 block multiply
    # U_tbl[s]: list of (i_local, j_local, coeff) for A block
    U_tbl = [
        [(0, 0, 1), (1, 1, 1)],       # P1: A00+A11
        [(1, 0, 1), (1, 1, 1)],       # P2: A10+A11
        [(0, 0, 1)],                   # P3: A00
        [(1, 1, 1)],                   # P4: A11
        [(0, 0, 1), (0, 1, 1)],       # P5: A00+A01
        [(1, 0, 1), (0, 0, -1)],      # P6: A10-A00
        [(0, 1, 1), (1, 1, -1)],      # P7: A01-A11
    ]
    # V_tbl[s]: list of (j_local, k_local, coeff) for B block
    V_tbl = [
        [(0, 0, 1), (1, 1, 1)],       # P1: B00+B11
        [(0, 0, 1)],                   # P2: B00
        [(0, 1, 1), (1, 1, -1)],      # P3: B01-B11
        [(1, 0, 1), (0, 0, -1)],      # P4: B10-B00
        [(1, 1, 1)],                   # P5: B11
        [(0, 0, 1), (0, 1, 1)],       # P6: B00+B01
        [(1, 0, 1), (1, 1, 1)],       # P7: B10+B11
    ]
    # W_tbl[s]: list of (i_local, k_local, coeff) for output C block
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    W_tbl = [
        [(0, 0, 1), (1, 1, 1)],       # P1: C00+=1, C11+=1
        [(1, 0, 1), (1, 1, -1)],      # P2: C10+=1, C11-=1
        [(0, 1, 1), (1, 1, 1)],       # P3: C01+=1, C11+=1
        [(0, 0, 1), (1, 0, 1)],       # P4: C00+=1, C10+=1
        [(0, 0, -1), (0, 1, 1)],      # P5: C00-=1, C01+=1
        [(1, 1, 1)],                   # P6: C11+=1
        [(0, 0, 1)],                   # P7: C00+=1
    ]

    col = 0
    for a in range(2):        # A column-block: j in {0,1} or {2,3}
        j_off = a * 2
        for b in range(3):    # B col-block: k in {0,1}, {2,3}, or {4,5}
            k_off = b * 2
            for s in range(7):  # Strassen term
                if _time.monotonic() - _t0 > _DEADLINE:
                    return {"U": U, "V": V, "W": W}

                # Fill U: row index = i * m + j
                for i_loc, j_loc, c in U_tbl[s]:
                    U[i_loc * m + (j_off + j_loc)][col] += c

                # Fill V: row index = j * p + k
                for j_loc, k_loc, c in V_tbl[s]:
                    V[(j_off + j_loc) * p + (k_off + k_loc)][col] += c

                # Fill W: row index = k * n + i
                for i_loc, k_loc, c in W_tbl[s]:
                    W[(k_off + k_loc) * n + i_loc][col] += c

                col += 1

    return {"U": U, "V": V, "W": W}
