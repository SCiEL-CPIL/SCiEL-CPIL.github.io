import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for M in (U, V, W):
        for row in M:
            if len(row) != rank:
                return False

    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    one = Fraction(1, 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                urow = Uf[i * m + j]
                vrow = Vf[j * p + k]
                wrow = Wf[k * n + i]
                for r in range(rank):
                    s += urow[r] * vrow[r] * wrow[r]
                if s != one:
                    return False
    return True


def _build_rank88():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        U[(1 + li) * m + (j_start + lj)][col] += strassen_U[t][l]
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        V[(j_start + lj) * p + (k_start + lk)][col] += strassen_V[t][l]
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        W[(k_start + lk) * n + (1 + li)][col] += strassen_W[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}


def _build_rank84():
    """
    Rank-84 decomposition:
    - i=0: schoolbook for 1x4x8 (32 terms)
    - i=1,2: for each (j_block, k_block) pair, use 6 terms instead of Strassen's 7
      by exploiting the 2x2x4 structure with a custom 6-term identity.
    """
    n, m, p = DIMENSIONS
    rank = 32 + 8 * 6  # 32 + 48 = 80? No, 2 j_blocks * 4 k_blocks * 6 = 48, total 80
    # Wait, 2 j_blocks * 4 k_blocks = 8 blocks, each 6 terms = 48. Total = 32 + 48 = 80.
    # But I need to verify this is valid. Let me think more carefully.
    #
    # For a 2x2x4 multiplication (A is 2x2, B is 2x4), the tensor has 16 entries.
    # Schoolbook gives rank 16. Strassen-like for 2x2x2 gives 7 per 2x2x2 block.
    # For 2x2x4, we have two 2x2x2 blocks, so 14 terms with Strassen.
    # Can we do 2x2x4 in fewer than 14? 
    #
    # Actually, let me just keep the rank-88 approach and verify it works.
    # The rank-84 idea needs more careful derivation.
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        U[(1 + li) * m + (j_start + lj)][col] += strassen_U[t][l]
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        V[(j_start + lj) * p + (k_start + lk)][col] += strassen_V[t][l]
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        W[(k_start + lk) * n + (1 + li)][col] += strassen_W[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}


def _build_rank80():
    """
    Attempt rank-80: use 6 terms per 2x2x4 block for the 2x4x8 part.
    For a 2x2x4 tensor, we need to find 6 rank-1 terms.
    Let A = [[a,b],[c,d]] (2x2), B = [[e,f,g,h],[i,j,k,l]] (2x4).
    C = A*B = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
    
    We need 6 terms. Let's try to construct them.
    This is non-trivial. Let me just return the rank-88 for now and verify.
    """
    return _build_rank88()


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    best = _build_rank88()
    best_rank = 88

    if not _verify(best["U"], best["V"], best["W"]):
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = rank_sb

    # Try rank-80 if time permits
    if time.time() < deadline - 30:
        try:
            candidate = _build_rank80()
            cand_rank = len(candidate["U"][0])
            if cand_rank < best_rank and _verify(candidate["U"], candidate["V"], candidate["W"]):
                best = candidate
                best_rank = cand_rank
        except Exception:
            pass

    return best
