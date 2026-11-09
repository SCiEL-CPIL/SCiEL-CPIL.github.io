import time
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time as _time
import numpy as np

def construct():
    _start = _time.time()
    _deadline = _start + 140
    n, m, p = DIMENSIONS  # all 6

    # Fallback: Rank 189 decomposition (3x3 grid of 2x2 Strassen blocks)
    def construct_rank189():
        strassen_U = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 1, 1), (1, 1, -1)],
        ]
        strassen_V = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
        ]
        strassen_W = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)],
        ]
        rank = 189
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for bi in range(3):
            for bk in range(3):
                for bj in range(3):
                    for t in range(7):
                        for (li, lk, coeff) in strassen_U[t]:
                            gi = 2 * bi + li
                            gk = 2 * bk + lk
                            U[gi * m + gk][col] += coeff
                        for (lk, lj, coeff) in strassen_V[t]:
                            gk = 2 * bk + lk
                            gj = 2 * bj + lj
                            V[gk * p + gj][col] += coeff
                        for (li, lj, coeff) in strassen_W[t]:
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            W[gj * n + gi][col] += coeff
                        col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    # Attempt Rank 184 decomposition (2x2 grid of 3x3 rank-23 blocks)
    # We need the rank-23 decomposition for 3x3 matrix multiplication.
    # If we can't find it, we return the rank-189 result.
    # For now, we return the rank-189 result to ensure validity.
    
    result = construct_rank189()
    
    # Verify and return
    if _time.time() > _deadline:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
        
    return {"U": result["U"], "V": result["V"], "W": result["W"]}

# EVOLVE-BLOCK-END
