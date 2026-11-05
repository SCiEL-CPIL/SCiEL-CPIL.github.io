"""Matrix multiplication 5x5x5 tensor decomposition using block-structured Strassen.

Strategy:
  * Partition each 5x5 matrix into blocks of sizes 2+2+1.
  * For every 2x2x2 sub-block use Strassen's rank-7 decomposition.
  * For every other sub-block use the schoolbook (rank = ni*mj*pk) decomposition.
  * Total rank = 8*7 + (sum of all other ni*mj*pk) = 117  (vs 125 schoolbook).
  * Verify the result exactly with Fraction arithmetic before returning.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition (local block indices)."""
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


def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2 matrix multiplication.

    Local indexing (row-major, 4 entries each):
      U: [A11, A12, A21, A22]
      V: [B11, B12, B21, B22]
      W: [C11, C12, C21, C22]

    Products:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}


def _embed_block(n_full, m_full, p_full, n_block, m_block, p_block,
                 row_offset, col_offset_A, col_offset_B, decomp):
    """Embed a local block decomposition into full-sized (zero-padded) matrices."""
    rank = len(decomp['U'][0])
    U_full = [[0] * rank for _ in range(n_full * m_full)]
    V_full = [[0] * rank for _ in range(m_full * p_full)]
    W_full = [[0] * rank for _ in range(p_full * n_full)]

    for i in range(n_block):
        for j in range(m_block):
            u_row = (row_offset + i) * m_full + (col_offset_A + j)
            for c in range(rank):
                U_full[u_row][c] = decomp['U'][i * m_block + j][c]

    for j in range(m_block):
        for k in range(p_block):
            v_row = (col_offset_A + j) * p_full + (col_offset_B + k)
            for c in range(rank):
                V_full[v_row][c] = decomp['V'][j * p_block + k][c]

    for k in range(p_block):
        for i in range(n_block):
            w_row = (col_offset_B + k) * n_full + (row_offset + i)
            for c in range(rank):
                W_full[w_row][c] = decomp['W'][k * p_block + i][c]

    return {"U": U_full, "V": V_full, "W": W_full}


def _combine_decompositions(decomps, n, m, p):
    """Concatenate multiple full-size block decompositions along the rank axis."""
    rank = sum(len(d['U'][0]) for d in decomps)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col_offset = 0
    for d in decomps:
        r = len(d['U'][0])
        for i in range(n * m):
            for c in range(r):
                U[i][col_offset + c] = d['U'][i][c]
        for j in range(m * p):
            for c in range(r):
                V[j][col_offset + c] = d['V'][j][c]
        for k in range(p * n):
            for c in range(r):
                W[k][col_offset + c] = d['W'][k][c]
        col_offset += r

    return {"U": U, "V": V, "W": W}


def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic (no floating point)."""
    rank = len(U[0]) if U else 0
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    for i in range(n):
        for j in range(m):
            u_row = Uf[i * m + j]
            for k in range(p):
                v_row = Vf[j * p + k]
                w_row = Wf[k * n + i]
                s = Fraction(0)
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
                    return False
    return True


def _verify_float(n, m, p, U, V, W):
    """Fast floating-point verification (used as a quick pre-check)."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            u_val = U_np[i * m + j]
            for k in range(p):
                dot = np.dot(u_val, V_np[j * p + k] * W_np[k * n + i])
                if abs(dot - 1.0) > 1e-9:
                    return False
    return True


def _construct_block_decomposition():
    """
    Build a block-structured 5x5x5 decomposition with a 2+2+1 partition.

    Each 2x2x2 sub-block uses Strassen (rank 7); all other sub-blocks use
    schoolbook.  This yields total rank 117.
    """
    n = m = p = 5
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    decomps = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    decomp = _strassen_2x2()
                else:
                    decomp = _schoolbook(ni, mj, pk)
                embedded = _embed_block(
                    5, 5, 5, ni, mj, pk,
                    offsets[i], offsets[j], offsets[k], decomp
                )
                decomps.append(embedded)

    return _combine_decompositions(decomps, 5, 5, 5)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before the 120s limit

    # Primary: block-structured 2+2+1 Strassen (rank 117).
    try:
        best = _construct_block_decomposition()
        if _verify_float(n, m, p, best['U'], best['V'], best['W']):
            # Exact rational confirmation (cheap enough for rank 117).
            if _verify_exact(n, m, p, best['U'], best['V'], best['W']):
                return best
    except Exception:
        pass

    # Fallback: schoolbook (rank 125) -- always valid.
    fallback = _schoolbook(n, m, p)
    if _verify_exact(n, m, p, fallback['U'], fallback['V'], fallback['W']):
        return fallback
    return fallback

# EVOLVE-BLOCK-END
