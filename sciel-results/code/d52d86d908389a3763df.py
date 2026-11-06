"""Matrix multiplication 6x6x6 — block-Strassen (2x2 blocks of 3x3) rank-189 decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# Deadline guard: kill well before 120s wall-clock limit
_START = time.time()
_DEADLINE = _START + 100.0  # 20s safety margin


def _check_deadline():
    if time.time() > _DEADLINE:
        raise TimeoutError("exceeded internal deadline")


def construct():
    """Return dict with keys 'U','V','W' whose tensors exactly decompose 6x6 matrix multiplication.

    Uses 2x2 block-Strassen at the block level (7 block multiplications of 3x3 matrices),
    each block multiplication decomposed via schoolbook (27 scalar products).
    Total rank = 7 * 27 = 189.

    Strassen block formulas:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)

      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    # Each 3x3 block multiplication has rank 9 (schoolbook).
    # 7 Strassen steps -> total rank 63? No: each step involves 3x3*3x3 which is 9 scalar products.
    # Actually each "block multiplication" M_s = X_s * Y_s where X_s, Y_s are 3x3 matrices.
    # The schoolbook decomposition of 3x3*3x3 has rank 27 (3*3*3).
    # 7 such decompositions -> rank = 7 * 27 = 189.

    rank = 189

    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189

    # Strassen coefficients: for each step s in 0..6,
    # alpha[s][a][b] = coefficient of A_{ab} in X_s
    # beta[s][c][d]  = coefficient of B_{cd} in Y_s
    # gamma[s][e][f] = coefficient of M_s in C_{ef}

    # A indices: (a,b) where a,b in {0,1}
    # B indices: (c,d) where c,d in {0,1}
    # C indices: (e,f) where e,f in {0,1}

    # alpha[s][a][b]: coefficient of A_{ab} in X_s
    #   s=0: M1 = (A00+A11)(B00+B11) -> X = A00+A11
    #   s=1: M2 = (A10+A11)B00 -> X = A10+A11
    #   s=2: M3 = A00(B01-B11) -> X = A00
    #   s=3: M4 = A11(B10-B00) -> X = A11
    #   s=4: M5 = (A00+A01)B11 -> X = A00+A01
    #   s=5: M6 = (A10-A00)(B00+B01) -> X = A10-A00
    #   s=6: M7 = (A01-A11)(B10+B11) -> X = A01-A11
    alpha = [
        {(0,0): 1, (1,1): 1},       # s=0
        {(1,0): 1, (1,1): 1},       # s=1
        {(0,0): 1},                  # s=2
        {(1,1): 1},                  # s=3
        {(0,0): 1, (0,1): 1},       # s=4
        {(1,0): 1, (0,0): -1},      # s=5
        {(0,1): 1, (1,1): -1},      # s=6
    ]

    # beta[s][c][d]: coefficient of B_{cd} in Y_s
    #   s=0: Y = B00+B11
    #   s=1: Y = B00
    #   s=2: Y = B01-B11
    #   s=3: Y = B10-B00
    #   s=4: Y = B11
    #   s=5: Y = B00+B01
    #   s=6: Y = B10+B11
    beta = [
        {(0,0): 1, (1,1): 1},       # s=0
        {(0,0): 1},                  # s=1
        {(0,1): 1, (1,1): -1},      # s=2
        {(1,0): 1, (0,0): -1},      # s=3
        {(1,1): 1},                  # s=4
        {(0,0): 1, (0,1): 1},       # s=5
        {(1,0): 1, (1,1): 1},       # s=6
    ]

    # gamma[s][e][f]: coefficient of M_s in C_{ef}
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    gamma = [
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1},   # s=0 (M1): C00+=1, C11+=1
        {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1},  # s=1 (M2): C10+=1, C11-=1
        {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1},   # s=2 (M3): C01+=1, C11+=1
        {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0},   # s=3 (M4): C00+=1, C10+=1
        {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0},  # s=4 (M5): C00-=1, C01+=1
        {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1},   # s=5 (M6): C11+=1
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},   # s=6 (M7): C00+=1
    ]

    # For each Strassen step s and each scalar product (u,v,w) in the 3x3 schoolbook:
    #   Row index in U: i_block*3 + row, where i_block in {0,1} from alpha
    #     A_{ab} is block at rows [3a..3a+2], cols [3b..3b+2]
    #     So the (u-th row, v-th col) of A_{ab} is at global (3a+u, 3b+v)
    #     U index = (3a+u)*6 + (3b+v)
    #
    #   Row index in V: (3c+v)*6 + (3d+w)  [B_{cd} is at rows 3c..3c+2, cols 3d..3d+2]
    #     The (v-th row, w-th col) of B_{cd} is at global (3c+v, 3d+w)
    #     V index = (3c+v)*6 + (3d+w)
    #
    #   Row index in W: (3e+w)*6 + (3f+u)  [C_{ef} is at rows 3e..3e+2, cols 3f..3f+2]
    #     The (w-th row, u-th col) of C_{ef} is at global (3e+w, 3f+u)
    #     W index = (3e+w)*6 + (3f+u)
    #
    #   The scalar product contributes alpha[s][ab] * beta[s][cd] to U and V,
    #   and gamma[s][ef] to W.

    col = 0
    for s in range(7):
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    for ab in alpha[s]:
                        a, b = ab
                        a_coeff = alpha[s][ab]
                        u_idx = (3 * a + u) * 6 + (3 * b + v)
                        for cd in beta[s]:
                            c, d = cd
                            b_coeff = beta[s][cd]
                            v_idx = (3 * c + v) * 6 + (3 * d + w)
                            U[u_idx][col] += a_coeff
                            V[v_idx][col] += b_coeff
                            for ef in gamma[s]:
                                e, f = ef
                                w_coeff = gamma[s][ef]
                                if w_coeff == 0:
                                    continue
                                w_idx = (3 * e + w) * 6 + (3 * f + u)
                                W[w_idx][col] += w_coeff
                    col += 1

    assert col == rank, f"column {col} != rank {rank}"

    _check_deadline()

    return {"U": U, "V": V, "W": W}
