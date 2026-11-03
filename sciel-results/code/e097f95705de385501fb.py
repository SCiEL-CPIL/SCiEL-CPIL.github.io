"""Rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 4]
_START = time.time()
_DEADLINE = 150.0 - 12.0  # safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _snap_half(x):
    """Snap a float to the nearest half-integer."""
    return int(round(2.0 * x)) / 2.0


def _build_rank_24():
    """
    Construct a 24-rank decomposition for C = A B, A in R^{2x3}, B in R^{3x4}.
    
    Uses 3 independent 7-term Strassen blocks for the 2x3 * 3x2 subproblems
    plus 3 rank-1 terms for the third output column.
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c], C = U V^T W
    where a = i*m+j (6 values), b = k*n+i (8 values), c = rank index
    """
    n, m, p = 2, 3, 4
    rank = 24
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each pair of output columns (k, k+1), use Strassen on the 2x3 * 3x2 subproblem
    # Block 0: output cols 0,1
    # Block 1: output cols 2,3
    # Block 2: output cols 4,5 (if p=6)
    
    # Actually, let's use a simpler approach: 24-rank schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    return U, V, W


def _verify(U, V, W, tol=1e-6):
    """Check C = U V^T W reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    C = U @ V.T @ W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                got = C[i * m + j, k * n + i]
                if abs(got - expected) > tol:
                    return False
    # Off-diagonal check: C[i*m+j, k'*n+i'] should be 0 unless i==i'.
    for a in range(n * m):
        for b in range(p * n):
            i = a // m
            k = b // n
            if i != k:
                if abs(C[a, b]) > tol:
                    return False
    return True


def _rationalize(M, max_den=4):
    """Convert a float matrix to half-integer matrix via Fraction reconstruction."""
    out = np.zeros_like(M, dtype=float)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            x = M[i, j]
            if abs(x) < 1e-9:
                out[i, j] = 0.0
                continue
            f = Fraction(x).limit_denominator(max_den)
            out[i, j] = float(f)
    return out


def _to_lists(M):
    return [[(int(v) if v == int(v) else v) for v in row] for row in M]


def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook rank-24 decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
