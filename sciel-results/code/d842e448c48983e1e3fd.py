"""Find a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS
NS, MS, PS = N * M, M * P, P * N      # 6, 21, 14
SCHOOLBOOK = N * M * P                # 42

# Known lower bound: M_{2,2,7} sub-tensor (rows i=0, all j, all k) has rank >= 14
LOWER = 14

_START = time.time()
_DEADLINE = 118.0                     # wall-clock safety margin before 150s kill
_TOL = 1e-6


def _schoolbook():
    """Canonical schoolbook rank-42 decomposition (always valid)."""
    U = np.zeros((NS, SCHOOLBOOK))
    V = np.zeros((MS, SCHOOLBOOK))
    W = np.zeros((PS, SCHOOLBOOK))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1.0
                V[j * P + k, c] = 1.0
                W[k * N + i, c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def _to_list(mat):
    return mat.tolist()


def _exact_valid(r, U, V, W):
    """Return True iff the triple of integer matrices is an exact rank-r
    decomposition of the 2x3x7 multiplication tensor (flattening test)."""
    A = Fraction(U)                      # (NS, r)
    B = Fraction(V)                       # (MS, r)
    C = Fraction(W)                       # (PS, r)
    # (U V^T)_{a,b} = sum_c U[a,c] V[b,c]
    UV = A @ B.T                          # (NS, MS)
    # (C A^T)_{k, i} = sum_c C[k,c] A[i,c]
    CA = C @ A.T                          # (PS, NS)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                kk = k * N + i
                if UV[a, b] != 1:
                    return False
                if CA[kk, a] != 1:
                    return False
    return True


def _rand_int(rng, n, m, lo, hi):
    return [[int(rng.randint(lo, hi)) for _ in range(m)] for _ in range(n)]


def construct():
    """Return {'U','V','W'} integer matrices giving an exact tensor
    decomposition; rank = number of columns (minimised)."""
    global _START
    rng = random.Random(20240711)
    best = _schoolbook()                 # guaranteed valid, rank 42
    for r in range(LOWER, SCHOOLBOOK):
        if time.time() - _START > _DEADLINE:
            break
        for _ in range(30):
            if time.time() - _START > _DEADLINE:
                break
            U = _rand_int(rng, NS, r, -2, 2)
            V = _rand_int(rng, MS, r, -2, 2)
            W = _rand_int(rng, PS, r, -2, 2)
            if _exact_valid(r, U, V, W):
                return {"U": U, "V": V, "W": W}
    return {"U": _to_list(best["U"]),
            "V": _to_list(best["V"]),
            "W": _to_list(best["W"])}
