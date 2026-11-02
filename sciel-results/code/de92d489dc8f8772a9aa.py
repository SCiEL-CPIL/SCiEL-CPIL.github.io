"""Strassen-based low-rank decomposition for 2x4x5 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_block(i0, j0, k0):
    """Rank-7 Strassen triples for a 2x2x2 block.
    Returns list of (u_dict, v_dict, w_dict).
    """
    def Ue(i, j, val): return (i0 + i, j0 + j), val
    def Ve(j, k, val): return (j0 + j, k0 + k), val
    def We(k, i, val): return (k0 + k, i0 + i), val

    M = [
        # M1 = (a00+a11)(b00+b11)
        ([(Ue(0,0,1), Ue(1,1,1))],
         [(Ve(0,0,1), Ve(1,1,1))],
         [(We(0,0,1), We(1,1,1))]),
        # M2 = (a10+a11)*b00
        ([(Ue(1,0,1), Ue(1,1,1))],
         [(Ve(0,0,1),)],
         [(We(0,1,1), We(1,1,-1))]),
        # M3 = a00*(b01-b11)
        ([(Ue(0,0,1),)],
         [(Ve(0,1,1), Ve(1,1,-1))],
         [(We(1,0,1), We(1,1,1))]),
        # M4 = a11*(b10-b00)
        ([(Ue(1,1,1),)],
         [(Ve(1,0,1), Ve(0,0,-1))],
         [(We(0,0,1), We(0,1,1))]),
        # M5 = (a00+a01)*b11
        ([(Ue(0,0,1), Ue(0,1,1))],
         [(Ve(1,1,1),)],
         [(We(0,0,-1), We(1,0,1))]),
        # M6 = (a10-a00)*(b00+b01)
        ([(Ue(1,0,1), Ue(0,0,-1))],
         [(Ve(0,0,1), Ve(0,1,1))],
         [(We(1,1,1),)]),
        # M7 = (a01-a11)*(b10+b11)
        ([(Ue(0,1,1), Ue(1,1,-1))],
         [(Ve(1,0,1), Ve(1,1,1))],
         [(We(0,0,1),)]),
    ]
    return [(dict(u), dict(v), dict(w)) for u, v, w in M]

def _schoolbook_block(i0, j0, k0):
    """Rank-4 schoolbook for a 2x2x1 block (row k0 only)."""
    blocks = []
    for i in range(2):
        for j in range(2):
            blocks.append((
                {(i0+i, j0+j): 1},
                {(j0+j, k0): 1},
                {(k0, i0+i): 1},
            ))
    return blocks

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2,4,5
    all_blocks = []
    all_blocks += _strassen_block(0, 0, 0)   # i 0-1, j 0-1, k 0-1
    all_blocks += _strassen_block(0, 2, 2)   # i 0-1, j 2-3, k 2-3
    all_blocks += _schoolbook_block(0, 0, 4) # i 0-1, j 0-1, k 4
    all_blocks += _schoolbook_block(0, 2, 4) # i 0-1, j 2-3, k 4
    rank = len(all_blocks)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (ud, vd, wd) in enumerate(all_blocks):
        for (i, j), val in ud.items():
            U[i * m + j][c] = val
        for (j, k), val in vd.items():
            V[j * p + k][c] = val
        for (k, i), val in wd.items():
            W[k * n + i][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
