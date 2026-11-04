"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 145

    def _verify_exact(U, V, W, n, m, p):
        """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if val != 1:
                        return False
        return True

    def _is_valid_entries(M):
        """Check all entries are integers or half-integers (denominator 1 or 2)."""
        for row in M:
            for val in row:
                if val.denominator not in (1, 2):
                    return False
        return True

    def _build_and_check(U_list, V_list, W_list, n, m, p):
        rank = len(U_list[0])
        U = np.array([[Fraction(v) for v in row] for row in U_list], dtype=object)
        V = np.array([[Fraction(v) for v in row] for row in V_list], dtype=object)
        W = np.array([[Fraction(v) for v in row] for row in W_list], dtype=object)
        if _is_valid_entries(U) and _is_valid_entries(V) and _is_valid_entries(W):
            if _verify_exact(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _strassen_block(j0, k0):
        """Return 7 Strassen terms for 2x2x2 block at inner-index j0, output-index k0."""
        terms = []
        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+0] = 1; u[1*3+1] = 1
        v[j0*5+k0] = 1; v[(j0+1)*5+(k0+1)] = 1
        w[k0*2+0] = 1; w[(k0+1)*2+1] = 1
        terms.append((u[:], v[:], w[:]))
        # M2 = (A10+A11)*B00 -> C10 += M2
        u = [0]*6; v = [0]*15; w = [0]*10
        u[1*3+0] = 1; u[1*3+1] = 1
        v[j0*5+k0] = 1
        w[k0*2+1] = 1
        terms.append((u[:], v[:], w[:]))
        # M3 = A00*(B10-B11) -> C00 += M3, C11 += M3
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+0] = 1
        v[(j0+1)*5+k0] = 1; v[(j0+1)*5+(k0+1)] = -1
        w[k0*2+0] = 1; w[(k0+1)*2+1] = 1
        terms.append((u[:], v[:], w[:]))
        # M4 = A11*(B01-B00) -> C00 += M4, C10 += M4
        u = [0]*6; v = [0]*15; w = [0]*10
        u[1*3+1] = 1
        v[j0*5+(k0+1)] = 1; v[j0*5+k0] = -1
        w[k0*2+0] = 1; w[k0*2+1] = 1
        terms.append((u[:], v[:], w[:]))
        # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+0] = 1; u[0*3+1] = 1
        v[(j0+1)*5+(k0+1)] = 1
        w[k0*2+0] = -1; w[(k0+1)*2+0] = 1
        terms.append((u[:], v[:], w[:]))
        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        u = [0]*6; v = [0]*15; w = [0]*10
        u[1*3+0] = 1; u[0*3+0] = -1
        v[j0*5+k0] = 1; v[j0*5+(k0+1)] = 1
        w[(k0+1)*2+1] = 1
        terms.append((u[:], v[:], w[:]))
        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+1] = 1; u[1*3+1] = -1
        v[(j0+1)*5+k0] = 1; v[(j0+1)*5+(k0+1)] = 1
        w[k0*2+0] = 1
        terms.append((u[:], v[:], w[:]))
        return terms

    # --- Known good: rank-28 via Strassen blocks + schoolbook ---
    def _rank28():
        terms = []
        for u, v, w in _strassen_block(0, 0):
            terms.append((u, v, w))
        for u, v, w in _strassen_block(0, 2):
            terms.append((u, v, w))
        # Schoolbook for j=0,1; k=4
        for i in range(2):
            for j in range(2):
                u = [0]*6; v = [0]*15; w = [0]*10
                u[i*3+j] = 1
                v[j*5+4] = 1
                w[4*2+i] = 1
                terms.append((u, v, w))
        # Schoolbook for j=2; k=0..4
        for i in range(2):
            for k in range(5):
                u = [0]*6; v = [0]*15; w = [0]*10
                u[i*3+2] = 1
                v[2*5+k] = 1
                w[k*2+i] = 1
                terms.append((u, v, w))
        U_list = [[t[0][i] for t in terms] for i in range(6)]
        V_list = [[t[1][i] for t in terms] for i in range(15)]
        W_list = [[t[2][i] for t in terms] for i in range(10)]
        return _build_and_check(U_list, V_list, W_list, n, m, p)

    # First, verify the known rank-28 works
    result = _rank28()
    if result is not None:
        return result

    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
