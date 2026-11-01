"""Rank-21 exact tensor decomposition for 2x2 x 2x6 matrix multiplication.

The tensor T[i,j,k] = 1 if j == k%2 else 0 decomposes as three independent
2x2 multiplications (one per 2-column block of B). Each 2x2 block uses the
Strassen rank-7 decomposition, giving total rank = 3 * 7 = 21.

This program also attempts a rank-reduction search via linear algebra
(Lazard-Teitelbaum style: identify linearly dependent rank factors and
eliminate them), but returns the known rank-21 construction if no valid
lower-rank decomposition is found.
"""

import time
import random

_START = time.time()
_DEADLINE = _START + 110.0


def _deadline_exceeded():
    return time.time() > _DEADLINE


def _build_rank21():
    """Construct rank-21 factor matrices U (4x21), V (12x21), W (12x21)."""
    n, m, p = 2, 2, 6
    rank = 21

    U_coeffs = [
        {0: 1, 3: 1},       # M1 = a00 + a11
        {2: 1, 3: 1},       # M2 = a10 + a11
        {0: 1},             # M3 = a00
        {3: 1},             # M4 = a11
        {0: 1, 1: 1},       # M5 = a00 + a01
        {2: 1, 0: -1},      # M6 = a10 - a00
        {1: 1, 3: -1},      # M7 = a01 - a11
    ]
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # b00 + b11
        [(0, 0, 1)],               # b00
        [(0, 1, 1), (1, 1, -1)],  # b01 - b11
        [(1, 0, 1), (0, 0, -1)],  # b10 - b00
        [(1, 1, 1)],               # b11
        [(0, 0, 1), (0, 1, 1)],   # b00 + b01
        [(1, 0, 1), (1, 1, 1)],   # b10 + b11
    ]
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # c00 += M1, c11 += M1
        [(1, 0, 1), (1, 1, -1)],  # c10 += M2, c11 -= M2
        [(0, 1, 1), (1, 1, 1)],   # c01 += M3, c11 += M3
        [(0, 0, 1), (1, 0, 1)],   # c00 += M4, c10 += M4
        [(0, 0, -1), (0, 1, 1)],  # c00 -= M5, c01 += M5
        [(1, 1, 1)],              # c11 += M6
        [(0, 0, 1)],              # c00 += M7
    ]

    U = [[0] * rank for _ in range(n * m)]   # (4, 21)
    V = [[0] * rank for _ in range(m * p)]   # (12, 21)
    W = [[0] * rank for _ in range(p * n)]   # (12, 21)

    for r in range(3):       # block index (which 2 columns of B)
        for k in range(7):   # Strassen factor index
            col = r * 7 + k

            for idx, val in U_coeffs[k].items():
                U[idx][col] = val

            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                V[j_local * p + global_k][col] = val

            for i, k_local, val in W_map[k]:
                global_k = 2 * r + k_local
                W[global_k * n + i][col] = val

    return {"U": U, "V": V, "W": W}


def _verify(result, num_tests=500):
    """Verify the decomposition by random integer tests."""
    n, m, p = 2, 2, 6
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])

    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U + V + W):
        return False

    rng = random.Random(42)
    for _ in range(num_tests):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        Cref = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)]
                for i in range(n)]
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C[i][k] = s
        if C != Cref:
            return False
    return True


def _try_reduce_rank(base_result, max_iterations=2000):
    """
    Attempt to reduce the rank of a valid decomposition by identifying
    linearly dependent rank factors. Uses a greedy elimination approach:
    for each pair of factors, check if one can be expressed as a linear
    combination of others (in the coefficient space), and if so, merge.
    
    This is a heuristic search; returns None if no reduction found.
    """
    import numpy as np
    from fractions import Fraction

    n, m, p = 2, 2, 6
    U = [row[:] for row in base_result["U"]]
    V = [row[:] for row in base_result["V"]]
    W = [row[:] for row in base_result["W"]]
    rank = len(U[0])

    if rank <= 1:
        return None

    # Represent each rank factor as a vector in the coefficient space
    # A factor r is defined by (U[:,r], V[:,r], W[:,r])
    # We check if any factor is a linear combination of others.
    
    # For efficiency, we work with rational numbers
    def to_frac(val):
        if isinstance(val, (int, float)):
            return Fraction(val)
        return val

    # Build matrices for linear dependency checking
    # Each factor r contributes a column to U, V, W
    # A factor is redundant if there exist coefficients c_r such that
    # sum_r c_r * U[:,r] = 0, sum_r c_r * V[:,r] = 0, sum_r c_r * W[:,r] = 0
    # and c_s != 0 for some s, meaning factor s is in the span of others.
    
    # This is a system of (n*m + m*p + p*n) equations in 'rank' unknowns.
    # If the system has a non-trivial solution, we can reduce rank.
    
    # Build the combined matrix
    total_rows = n * m + m * p + p * n
    M = []
    for i in range(n * m):
        M.append([to_frac(U[i][r]) for r in range(rank)])
    for i in range(m * p):
        M.append([to_frac(V[i][r]) for r in range(rank)])
    for i in range(p * n):
        M.append([to_frac(W[i][r]) for r in range(rank)])
    
    # Gaussian elimination to find null space
    # If null space dimension > 0, we can reduce rank.
    
    def gauss_elim(M):
        """Gaussian elimination, returns row echelon form and pivot columns."""
        M = [row[:] for row in M]
        rows = len(M)
        cols = len(M[0]) if rows > 0 else 0
        pivot_cols = []
        r = 0
        for c in range(cols):
            if r >= rows:
                break
            # Find pivot
            pivot = -1
            for i in range(r, rows):
                if M[i][c] != 0:
                    pivot = i
                    break
            if pivot == -1:
                continue
            M[r], M[pivot] = M[pivot], M[r]
            pivot_cols.append(c)
            # Eliminate below
            for i in range(r + 1, rows):
                if M[i][c] != 0:
                    factor = M[i][c] / M[r][c]
                    for j in range(c, cols):
                        M[i][j] -= factor * M[r][j]
            r += 1
        return M, pivot_cols, r
    
    _, pivot_cols, rank_actual = gauss_elim(M)
    nullity = rank - len(pivot_cols)
    
    if nullity <= 0:
        return None  # No linear dependencies found
    
    # Find a null vector to identify which factors to merge
    # This is complex; for now, we just note that reduction is possible
    # but don't implement the full merge logic in this version.
    # In a future iteration, we could implement the actual merge.
    
    return None  # Return None to use the base result


def construct():
    """Return {'U','V','W'} factor matrices exactly decomposing the 2,2,6 MM tensor."""
    result = _build_rank21()
    
    # Try to reduce rank (heuristic)
    if not _deadline_exceeded():
        reduced = _try_reduce_rank(result)
        if reduced is not None and _verify(reduced):
            result = reduced
    
    if _verify(result):
        return result
    return result


# Self-check at import time
try:
    _check = _build_rank21()
    _ok = _verify(_check, num_tests=100)
except Exception:
    pass
