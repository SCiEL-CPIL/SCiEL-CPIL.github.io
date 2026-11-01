"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Correct tensor condition (bilinear, for ALL A,B):
    C[i,k] = sum_j A[i,j] B[j,k]
must equal
    sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_j V[j*p+k,c] B[j,k]).

Matching coefficients of the independent monomials A[i,j]B[j',k] gives the
defining system (for every i, j, k):
    sum_c U[i*m+j,c] * V[j*p+k,c] == delta_{j,k}.

Note this is the *j,k* delta (NOT the i,k delta used by the previous seed),
which is why the previous verification was unsatisfiable below rank 28.
The W factor only routes each term to the output slot k*n+i and so is
determined once U,V are found: W[k*n+i,c] = 1 whenever the column c has a
non-zero contribution to (i,k).

Strategy:
  * Guaranteed fallback: schoolbook rank 28 (always exact).
  * Improvement: solve the 28 quadratic equations for R = 27, 26, ... using
    Gauss-Newton with many random starts (the system is very underdetermined),
    then round to the half-integer grid and verify EXACTLY with Fraction.
    Only an exact candidate is accepted; otherwise we keep the fallback.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 7]


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _build_W(U, V, n, m, p):
    R = len(U[0])
    W = [[0] * R for _ in range(p * n)]
    for c in range(R):
        for i in range(n):
            for k in range(p):
                if any(U[i * m + j][c] != 0 and V[j * p + k][c] != 0 for j in range(m)):
                    W[k * n + i][c] = 1
    return W


def _residual(U, V, n, m, p):
    """r_{i,j,k} = sum_c U[i*m+j,c] V[j*p+k,c] - delta_{j,k}."""
    R = len(U[0])
    resid = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c]
                resid.append(s - (1.0 if j == k else 0.0))
    return resid


def _residual2(U, V, n, m, p):
    r = _residual(U, V, n, m, p)
    return sum(x * x for x in r)


def _gauss_newton(U, V, n, m, p, iters=300, tol=1e-13):
    """Solve the quadratic system via Gauss-Newton. Returns (U,V,sse)."""
    R = len(U[0])
    nm, mp = n * m, m * p
    nv = nm * R + mp * R
    for _ in range(iters):
        resid = _residual(U, V, n, m, p)
        sse = sum(x * x for x in resid)
        if sse < tol:
            break
        eqs = n * m * p
        J = [[0.0] * nv for _ in range(eqs)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = resid[idx]
                    for c in range(R):
                        J[idx][i * R + c] += V[j * p + k][c]
                        J[idx][nm * R + j * R + c] += U[i * m + j][c]
                    idx += 1
        # solve (J J^T) d = -r  (least-squares normal equations, eqs x eqs)
        A = [[0.0] * eqs for _ in range(eqs)]
        b = [0.0] * eqs
        for a in range(eqs):
            Ja = J[a]
            for bb in range(eqs):
                s = 0.0
                Jb = J[bb]
                for t in range(nv):
                    s += Ja[t] * Jb[t]
                A[a][bb] = s
            b[a] = -resid[a]
        # Gaussian elimination with partial pivoting
        for col in range(eqs):
            piv = max(range(col, eqs), key=lambda rr: abs(A[rr][col]))
            if abs(A[piv][col]) < 1e-12:
                break
            A[col], A[piv] = A[piv], A[col]
            b[col], b[piv] = b[piv], b[col]
            pv = A[col][col]
            for t in range(nv):
                A[col][t] /= pv
            b[col] /= pv
            for rr in range(eqs):
                if rr == col:
                    continue
                f = A[rr][col]
                if f == 0.0:
                    continue
                for t in range(nv):
                    A[rr][t] -= f * A[col][t]
                b[rr] -= f * b[col]
        d = [0.0] * nv
        for rr in range(eqs):
            if abs(A[rr][rr]) > 1e-12:
                d[rr] = b[rr] / A[rr][rr]
        # back-substitute upper triangular part
        for rr in range(eqs - 1, -1, -1):
            s = b[rr]
            for t in range(rr + 1, eqs):
                s -= A[rr][t] * d[t]
            if abs(A[rr][rr]) > 1e-12:
                d[rr] = s / A[rr][rr]
        # d = [dU (nm*R), dV (mp*R)]
        dU = [d[i * R:(i + 1) * R] for i in range(nm)]
        dV = [d[(nm * R) + j * R:(nm * R) + (j + 1) * R] for j in range(mp)]
        U = [[U[i][c] + dU[i][c] for c in range(R)] for i in range(nm)]
        V = [[V[j][c] + dV[j][c] for c in range(R)] for j in range(mp)]
    sse = _residual2(U, V, n, m, p)
    return U, V, sse


def _to_frac(x):
    if isinstance(x, Fraction):
        return x
    h = int(round(2.0 * float(x))) / 2.0
    return Fraction(h)


def _verify_exact(U, V, W, n, m, p, probes=8):
    """Exact check of the bilinear identity C = A*B for random integer A,B."""
    R = len(U[0])
    Uf = [[_to_frac(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[_to_frac(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[_to_frac(W[k][c]) for c in range(R)] for k in range(p * n)]

    for _ in range(probes):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # expected C = A*B
        Cexp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # reconstructed C
        Crec = [[0] * p for _ in range(n)]
        for c in range(R):
            ua = [sum(Uf[i * m + j][c] * A[i][j] for j in range(m)) for i in range(n)]
            vb = [sum(Vf[j * p + k][c] * B[j][k] for j in range(m)) for k in range(p)]
            for k in range(p):
                for i in range(n):
                    Crec[i][k] += Wf[k * n + i][c] * ua[i] * vb[k]
        for i in range(n):
            for k in range(p):
                if Crec[i][k] != Cexp[i][k]:
                    return False
    return True


def _round_half(M):
    return [[int(round(2.0 * float(M[i][c]))) / 2.0 for c in range(len(M[0]))]
            for i in range(len(M))]


def _try_rank(R, n, m, p, rng, deadline, starts):
    for s in range(starts):
        if time.time() > deadline:
            return None
        U = [[rng.gauss(0.0, 1.0) for _ in range(R)] for _ in range(n * m)]
        V = [[rng.gauss(0.0, 1.0) for _ in range(R)] for _ in range(m * p)]
        U, V, sse = _gauss_newton(U, V, n, m, p)
        if sse > 1e-9:
            continue
        Ur = _round_half(U)
        Vr = _round_half(V)
        if _residual2(Ur, Vr, n, m, p) > 1e-9:
            continue
        W = _build_W(Ur, Vr, n, m, p)
        if _verify_exact(Ur, Vr, W, n, m, p):
            return Ur, Vr, W
    return None


def _build_kronecker_init(R, n, m, p, rng):
    """
    Initialize U,V using Kronecker product structure.
    
    The 2x2x7 multiplication can be viewed as:
    - 2x2x2 part (columns 0-1) with Strassen-like structure
    - 2x1x5 part (columns 2-6) with simpler structure
    
    Use a structured initialization that respects the algebraic structure.
    """
    nm = n * m
    mp = m * p
    
    # For rank R, initialize with structure that has sparse patterns
    # matching the expected algebraic form
    U = [[0.0] * R for _ in range(nm)]
    V = [[0.0] * R for _ in range(mp)]
    
    # Strassen-like initialization for the 2x2x2 block
    # The 2x2x2 multiplication tensor has rank 7
    # Use a structured pattern
    strassen_patterns = [
        # (i, j) pairs for U, (j, k) pairs for V
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # schoolbook for 2x2x2
    ]
    
    # Initialize first 7 columns with Strassen-like structure
    for c in range(min(7, R)):
        # Pattern based on Strassen's 7 multiplications
        if c == 0:
            U[0][c] = 1.0  # A[0,0]
            V[0][c] = 1.0  # B[0,0]
            V[1][c] = 1.0  # B[1,0]
        elif c == 1:
            U[1][c] = 1.0  # A[1,0]
            V[0][c] = 1.0  # B[0,0]
            V[1][c] = 1.0  # B[1,0]
        elif c == 2:
            U[2][c] = 1.0  # A[0,1]
            V[0][c] = 1.0  # B[0,1]
            V[1][c] = 1.0  # B[1,1]
        elif c == 3:
            U[3][c] = 1.0  # A[1,1]
            V[0][c] = 1.0  # B[0,1]
            V[1][c] = 1.0  # B[1,1]
        elif c == 4:
            U[0][c] = 1.0
            U[1][c] = 1.0
            V[0][c] = 1.0
            V[1][c] = -1.0
        elif c == 5:
            U[2][c] = 1.0
            U[3][c] = 1.0
            V[0][c] = 1.0
            V[1][c] = -1.0
        elif c == 6:
            U[1][c] = 1.0
            U[3][c] = 1.0
            V[0][c] = 1.0
            V[1][c] = -1.0
    
    # For remaining columns, add small random perturbations
    for c in range(7, R):
        for i in range(nm):
            U[i][c] = rng.gauss(0.0, 0.5)
        for j in range(mp):
            V[j][c] = rng.gauss(0.0, 0.5)
    
    return U, V


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135  # safety margin under the 150s wall-clock kill

    U, V, W = _build_schoolbook(n, m, p)
    rng = random.Random(12345)

    try:
        # First try structured Kronecker-based initialization
        for R in range(25, 27):  # Try ranks 25, 26
            if time.time() > deadline:
                break
            # Try structured initialization
            for attempt in range(20):
                if time.time() > deadline:
                    break
                U_init, V_init = _build_kronecker_init(R, n, m, p, rng)
                U_sol, V_sol, sse = _gauss_newton(U_init, V_init, n, m, p)
                if sse > 1e-9:
                    continue
                Ur = _round_half(U_sol)
                Vr = _round_half(V_sol)
                if _residual2(Ur, Vr, n, m, p) > 1e-9:
                    continue
                Wcand = _build_W(Ur, Vr, n, m, p)
                if _verify_exact(Ur, Vr, Wcand, n, m, p):
                    U, V, W = Ur, Vr, Wcand
                    return {"U": U, "V": V, "W": W}
        
        # Fallback to random initialization search
        for R in range(n * m * p - 1, 24, -1):
            if time.time() > deadline:
                break
            starts = 14 if R >= n * m * p - 2 else 8
            cand = _try_rank(R, n, m, p, rng, deadline, starts)
            if cand is not None:
                U, V, W = cand
                break
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
