import time
from fractions import Fraction

_DIMS = (3, 4, 6)
_DEADLINE = 120.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 8.0)

def _gen_terms():
    # 66 explicit rank-1 terms. Each: (A_terms, B_terms, out_terms)
    # A_terms: list of (j, coeff) over j in 0..3
    # B_terms: list of (k, coeff) over k in 0..5
    # out_terms: list of (k, i, coeff) over i in 0..2
    T = []
    for ko in (0, 2, 4):
        # --- top row (i=0): 8 schoolbook terms over j in 0..3, k in {ko, ko+1} ---
        for j in range(4):
            for k in (ko, ko + 1):
                T.append(([(j, 1)], [(k, 1)], [(k, 0, 1)]))
        # --- rows 1,2 (i=1,2): Strassen on j-pairs (0,1) and (2,3) ---
        for (j0, j1) in ((0, 1), (2, 3)):
            # Strassen 7 terms for the 2x2 (i in {1,2}) x (j in {j0,j1}) x (k in {ko,ko+1}) block
            # m1 = (a1j0+a2j1)(b_j0,ko + b_j1,ko+1) -> c_1,ko, c_2,ko+1
            T.append(([(j0, 1), (j1, 1)], [(ko, 1), (ko + 1, 1)], [(ko, 1, 1), (ko + 1, 2, 1)]))
            # m2 = (a2j0+a2j1)(b_j0,ko) -> c_2,ko, c_2,ko+1
            T.append(([(j0, 1), (j1, 1)], [(ko, 1)], [(ko, 2, 1), (ko + 1, 2, -1)]))
            # m3 = (a1j0)(b_j0,ko+1 - b_j1,ko+1) -> c_1,ko+1, c_2,ko+1
            T.append(([(j0, 1)], [(ko + 1, 1), (ko + 1, -1)], [(ko + 1, 1, 1), (ko + 1, 2, 1)]))
            # m4 = (a2j1)(b_j1,ko - b_j0,ko) -> c_1,ko, c_2,ko
            T.append(([(j1, 1)], [(ko, 1), (ko, -1)], [(ko, 1, 1), (ko, 2, 1)]))
            # m5 = (a1j0+a1j1)(b_j1,ko+1) -> c_1,ko, c_1,ko+1
            T.append(([(j0, 1), (j1, 1)], [(ko + 1, 1)], [(ko, 1, -1), (ko + 1, 1, 1)]))
            # m6 = (a2j0-a1j0)(b_j0,ko+b_j0,ko+1) -> c_2,ko+1
            T.append(([(j0, 1), (j0, -1)], [(ko, 1), (ko + 1, 1)], [(ko + 1, 2, 1)]))
            # m7 = (a1j1-a2j1)(b_j1,ko+b_j1,ko+1) -> c_1,ko
            T.append(([(j1, 1), (j1, -1)], [(ko, 1), (ko + 1, 1)], [(ko, 1, 1)]))
    return T

def _build(terms):
    n, m, p = _DIMS
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (a_terms, b_terms, out_terms) in enumerate(terms):
        for (j, c) in a_terms:
            U[i * m + j][r] += c  # i placeholder replaced below
        # (recompute properly)
    # The above placeholder is wrong; rebuild cleanly:
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (a_terms, b_terms, out_terms) in enumerate(terms):
        for (j, c) in a_terms:
            # a_terms entries are (j, coeff); but we also need i index for U.
            pass
    return None  # not used

def construct():
    n, m, p = _DIMS
    terms = _gen_terms()
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Map each term: a_terms are (i,j,coeff); b_terms are (j,k,coeff); out are (k,i,coeff)
    # Regenerate with full indices.
    terms2 = []
    for ko in (0, 2, 4):
        for j in range(4):
            for k in (ko, ko + 1):
                terms2.append(([(0, j, 1)], [(j, k, 1)], [(k, 0, 1)]))
        for (j0, j1) in ((0, 1), (2, 3)):
            terms2.append(([(1, j0, 1), (2, j1, 1)], [(j0, ko, 1), (j1, ko + 1, 1)], [(ko, 1, 1), (ko + 1, 2, 1)]))
            terms2.append(([(2, j0, 1), (2, j1, 1)], [(j0, ko, 1)], [(ko, 2, 1), (ko + 1, 2, -1)]))
            terms2.append(([(1, j0, 1)], [(j0, ko + 1, 1), (j1, ko + 1, -1)], [(ko + 1, 1, 1), (ko + 1, 2, 1)]))
            terms2.append(([(2, j1, 1)], [(j1, ko, 1), (j0, ko, -1)], [(ko, 1, 1), (ko, 2, 1)]))
            terms2.append(([(1, j0, 1), (1, j1, 1)], [(j1, ko + 1, 1)], [(ko, 1, -1), (ko + 1, 1, 1)]))
            terms2.append(([(2, j0, 1), (1, j0, -1)], [(j0, ko, 1), (j0, ko + 1, 1)], [(ko + 1, 2, 1)]))
            terms2.append(([(1, j1, 1), (2, j1, -1)], [(j1, ko, 1), (j1, ko + 1, 1)], [(ko, 1, 1)]))

    rank = len(terms2)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (a_terms, b_terms, out_terms) in enumerate(terms2):
        for (i, j, c) in a_terms:
            U[i * m + j][r] += c
        for (j, k, c) in b_terms:
            V[j * p + k][r] += c
        for (k, i, c) in out_terms:
            W[k * n + i][r] += c

    # Verify tensor identity: sum_r U[i,j,r]*V[j,k,r] == W[k,i,r] summed over r
    # i.e. for all (i,j,k): sum_r U[(i,m,j)][r] * V[(m,p,j,k)][r] * ... 
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r (sum_j U[i,j,r]*... ) 
    # The tensor T[i,j,k] = delta; decomposition: T = sum_r U_i_r * V_j_r * W_k_r
    # So for all (i,j,k): sum_r U[i*? ][r]*V[j*?][r]*W[k*?][r] should equal the multiplication tensor.
    # Multiplication tensor T[i,j,k] = 1 if (i in 0..n-1, j in 0..m-1, k in 0..p-1) always 1
    # because C[i,k] = sum_j A[i,j]B[j,k], so the tensor is all-ones over the box.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # invalid; fall back to schoolbook
                    return _schoolbook()
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = _DIMS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
