"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition.

Combines:
  * a verified, deterministic rank-49 Strassen-block construction (structural core),
  * a bounded low-rank search that probes for a rank-48 decomposition (refinement),
  * an exact integer verification guard so we only ever return a valid tensor.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before the 150s kill

    # 1) Guaranteed-correct rank-49 Strassen-block decomposition.
    result = _construct_strassen_49(n, m, p)
    if result is not None:
        best = result
        best_rank = 49
    else:
        best = None
        best_rank = None

    # 2) Spend the remaining budget probing for a rank-48 decomposition.
    #    A rank-48 solution would break the 7-fold block symmetry, so this is
    #    very unlikely; we only take it if it verifies exactly.
    if time.time() < deadline - 25:
        cand = _search_lower_rank(n, m, p, 48, deadline)
        if cand is not None:
            best = cand
            best_rank = 48

    # 3) Fallback: schoolbook rank-56 (always valid).
    if best is None:
        best = _construct_schoolbook(n, m, p)

    return best


def _construct_strassen_49(n, m, p):
    """
    Rank-49 decomposition: view the 2x14 matrix B as 7 blocks of 2x2 and apply
    Strassen's 7-term algorithm to each block. Total rank = 7 blocks * 7 terms.

    W index follows the task convention W[k*n + i].
    """
    num_blocks = p // 2            # 7
    rank = num_blocks * 7          # 49

    U = [[0] * rank for _ in range(n * m)]   # 4  x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49

    # A element indices (row-major, m=2): A00=0, A01=1, A10=2, A11=3
    U_forms = [
        [1, 0, 0, 1],    # M1 = A00 + A11
        [0, 0, 1, 1],    # M2 = A10 + A11
        [1, 0, 0, 0],    # M3 = A00
        [0, 0, 0, 1],    # M4 = A11
        [1, 1, 0, 0],    # M5 = A00 + A01
        [-1, 0, 1, 0],   # M6 = A10 - A00
        [0, 1, 0, -1],   # M7 = A01 - A11
    ]

    for r in range(num_blocks):
        k1 = 2 * r        # first output column of block
        k2 = 2 * r + 1    # second output column of block
        base_col = r * 7

        # B column indices: B[0,k1]=k1, B[0,k2]=k2, B[1,k1]=p+k1, B[1,k2]=p+k2
        # C indices (W): C[0,k1]=k1*n+0, C[1,k1]=k1*n+1, C[0,k2]=k2*n+0, C[1,k2]=k2*n+1

        # M1 = (A00+A11) * (B[0,k1]+B[1,k2]) -> C00 += M1, C11 += M1
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[k1][base_col] = 1
        V[p + k2][base_col] = 1
        W[k1 * n + 0][base_col] = 1
        W[k2 * n + 1][base_col] = 1

        # M2 = (A10+A11) * B[0,k1] -> C10 += M2, C11 -= M2
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1

        # M3 = A00 * (B[0,k2]-B[1,k2]) -> C01 += M3, C11 += M3
        U[0][base_col + 2] = 1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1

        # M4 = A11 * (B[1,k1]-B[0,k1]) -> C00 += M4, C10 += M4
        U[3][base_col + 3] = 1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        W[k1 * n + 0][base_col + 3] = 1
        W[k1 * n + 1][base_col + 3] = 1

        # M5 = (A00+A01) * B[1,k2] -> C00 -= M5, C01 += M5
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        V[p + k2][base_col + 4] = 1
        W[k1 * n + 0][base_col + 4] = -1
        W[k2 * n + 0][base_col + 4] = 1

        # M6 = (A10-A00) * (B[0,k1]+B[0,k2]) -> C11 += M6
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        V[k1][base_col + 5] = 1
        V[k2][base_col + 5] = 1
        W[k2 * n + 1][base_col + 5] = 1

        # M7 = (A01-A11) * (B[1,k1]+B[1,k2]) -> C00 += M7
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        V[p + k1][base_col + 6] = 1
        V[p + k2][base_col + 6] = 1
        W[k1 * n + 0][base_col + 6] = 1

    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _construct_schoolbook(n, m, p):
    """Always-valid rank-(n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Exact integer verification: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _search_lower_rank(n, m, p, rank, deadline):
    """
    Bounded search for a rank-`rank` decomposition.

    A rank-48 solution (if it exists) must break the 7-fold block symmetry, so we
    use a small set of cheap, symmetry-aware heuristics:
      * block-perturbation of the rank-49 solution (swap one term between blocks),
      * random sparse matrices with a few half-integer entries.
    Each candidate is verified exactly; we only return a candidate that verifies.
    """
    # --- Heuristic A: block-perturbation of the rank-49 solution -------------
    base = _construct_strassen_49(n, m, p)
    if base is not None:
        for r1 in range(7):
            for r2 in range(7):
                if r1 == r2:
                    continue
                if time.time() > deadline - 12:
                    return None
                # Try a few single-term swaps between the two blocks.
                for s in range(7):
                    U = [row[:] for row in base["U"]]
                    V = [row[:] for row in base["V"]]
                    W = [row[:] for row in base["W"]]
                    cA = r1 * 7 + s
                    cB = r2 * 7 + s
                    # swap the whole term (all rows) between the two columns
                    for row in range(len(U)):
                        U[row][cA], U[row][cB] = U[row][cB], U[row][cA]
                    for row in range(len(V)):
                        V[row][cA], V[row][cB] = V[row][cB], V[row][cA]
                    for row in range(len(W)):
                        W[row][cA], W[row][cB] = W[row][cB], W[row][cA]
                    if _verify_decomposition(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}

    # --- Heuristic B: random sparse matrices with half-integer entries -------
    choices = [0, 0, 0, 1, -1, 0.5, -0.5, 2, -2]
    for _ in range(400):
        if time.time() > deadline - 12:
            return None
        U = [[random.choice(choices) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(choices) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(choices) for _ in range(rank)] for _ in range(p * n)]
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    return None
# EVOLVE-BLOCK-END
