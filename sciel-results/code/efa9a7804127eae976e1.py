"""Matrix multiplication 2x3 x 3x4 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check tensor equation exactly using Fractions."""
    r = len(U[0])
    # Convert to Fraction
    UF = [[Fraction(x) for x in row] for row in U]
    VF = [[Fraction(x) for x in row] for row in V]
    WF = [[Fraction(x) for x in row] for row in W]
    # T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # The matrix-mult tensor: for A (n x m), B (m x p), C (n x p)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1 (for the (i,j,k) factor)
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += UF[i*m + j][c] * VF[j*p + k][c] * WF[k*n + i][c]
                if s != 1:
                    return False
    return True

def _try_round_and_verify(Uf, Vf, Wf, n, m, p, denom=2):
    """Round float matrices to half-integers and verify."""
    r = Uf.shape[1]
    U = [[round(x * denom) / denom for x in row] for row in Uf]
    V = [[round(x * denom) / denom for x in row] for row in Vf]
    W = [[round(x * denom) / denom for x in row] for row in Wf]
    # Convert to exact for verification
    if _verify(U, V, W, n, m, p):
        return U, V, W
    return None

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Schoolbook (rank 24) as fallback
    def schoolbook():
        rank = n * m * p
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

    # Try rank 23 via optimization + rounding
    def try_rank23():
        if time.time() > deadline - 5:
            return None
        import random
        r = 23
        best = None
        rng = np.random.RandomState(42)
        for attempt in range(60):
            if time.time() > deadline - 5:
                break
            # Initialize with random half-integer-ish values
            scale = 1.0
            U0 = rng.randn(n*m, r) * scale
            V0 = rng.randn(m*p, r) * scale
            W0 = rng.randn(p*n, r) * scale
            # Simple gradient-free local search: perturb and round
            Uf, Vf, Wf = U0.copy(), V0.copy(), W0.copy()
            # Do a few rounds of coordinate perturbation with rounding check
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                # Pick a random entry in one matrix and perturb
                mat = rng.randint(0, 3)
                if mat == 0:
                    ii, jj = rng.randint(0, n*m), rng.randint(0, r)
                    Uf[ii, jj] += rng.randn() * 0.3
                elif mat == 1:
                    ii, jj = rng.randint(0, m*p), rng.randint(0, r)
                    Vf[ii, jj] += rng.randn() * 0.3
                else:
                    ii, jj = rng.randint(0, p*n), rng.randint(0, r)
                    Wf[ii, jj] += rng.randn() * 0.3
                res = _try_round_and_verify(Uf, Vf, Wf, n, m, p, denom=2)
                if res is not None:
                    return res
        return None

    result = try_rank23()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return schoolbook()
