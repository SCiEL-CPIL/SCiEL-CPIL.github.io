"""Structural search for a rank-minimal exact tensor decomposition of 3x4x8 matrix mult."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 8]
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _rationalize(mat, tol=1e-9):
    """Convert a float matrix to a list of lists of Fractions (half-integers allowed)."""
    out = []
    for row in mat:
        r = []
        for x in row:
            if abs(x) < tol:
                r.append(Fraction(0))
                continue
            # try nearest half-integer
            h = round(2 * x) / 2
            if abs(x - h) < tol:
                r.append(Fraction(h).limit_denominator(1000))
            else:
                r.append(Fraction(x).limit_denominator(1000000))
        out.append(r)
    return out


def _check(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] == delta for C[i,k] coupling.
    The multiplication tensor T[i,j,k] (flattened as C[i,k] = sum_j A[i,j]B[j,k]) requires:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... 
    Actually the standard flattening: the bilinear map is
      C[i,k] = sum_{j,c} A[i,j] * (U? ) ...
    We define: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (contribution).
    The rank-r decomposition: there exist U (n*m x r), V (m*p x r), W (p*n x r) such that
      for all i,k:  sum_j A[i,j] B[j,k]  =  sum_c  (sum_j U[i*m+j,c] A[i,j])? 
    No. Correct form: C[i,k] = sum_c  U_c(A) * V_c(B)  where U_c(A) = sum_{i,j} U[i*m+j,c] A[i,j],
    V_c(B) = sum_{j,k} V[j*p+k,c] B[j,k], and the output is weighted by W? 
    Standard: T = sum_c u_c ⊗ v_c ⊗ w_c with u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n},
    and the multiplication satisfies: C[i,k] = sum_{j} A[i,j] B[j,k] = sum_c (sum_j u_c[i*m+j] A[i,j]) * (sum_j v_c[j*p+k] B[j,k]) * w_c[k*n+i]?? 
    That's not right either; w_c must be 1 on the matching (i,k) and the u,v must pick out the right entries.
    The exact identity (trace convention W[k*n+i, r]) is:
      sum_{j, r} A[i,j] B[j,k] = sum_r  (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k])  *  W[k*n+i, r]
    is NOT generally correct; the correct bilinear tensor identity is:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]   (as a 3-tensor in indices (i,j),(j,k),(k,i))
    i.e. the (i,j,k) entry of the multiplication tensor equals sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r].
    The multiplication tensor entry T[i,j,k] = 1 if the indices are consistent (always, since it's the
    canonical tensor) ... actually the matrix-mult tensor is T[i,j,k] = delta (the structure constant).
    For matrix mult, the tensor is T_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k (it's the "all-ones"
    coupling in the sense that C[i,k] = sum_j A[i,j]B[j,k]). So we require:
      sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1   for all i in [n], j in [m], k in [p].
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    # build via einsum over r
    # T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # vectorize: loop over i,k and check sum over j of (sum_r U[i*m+j,r]V[j*p+k,r]W[k*n+i,r]) == 1
    # but the identity is per (i,j,k), not summed over j. The tensor entry at (i,j,k) must equal 1.
    # So for every (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1.
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(r):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
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


def _try_rank(r, n, m, p, iters, seed):
    """Attempt to find a rank-r decomposition by random U,V and solving for W via least squares,
    then verify. Uses the identity sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k).
    For fixed U,V, the equations are linear in W (each equation involves W[k*n+i,c] linearly).
    We set up a linear system: for each (i,j,k), sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1.
    Note W[k*n+i,c] appears in equations for all j. So for fixed (i,k), the unknowns are
    W[k*n+i,0..r-1] and the equations over j give: sum_j? No, each j is a separate equation:
      for each j: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    So for fixed (i,k), we have m equations in r unknowns (W[k*n+i,c]). We can solve each (i,k)
    block independently! That's a big structural simplification.
    """
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if _elapsed() > _DEADLINE - 5.0:
            return None
        U = rng.integers(-1, 2, size=(n * m, r))
        V = rng.integers(-1, 2, size=(m * p, r))
        W = np.zeros((p * n, r))
        ok = True
        for i in range(n):
            for k in range(p):
                # unknowns W[k*n+i, c], c in 0..r-1
                # equations over j in 0..m-1: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
                Aeq = np.array([[U[i * m + j, c] * V[j * p + k, c] for c in range(r)] for j in range(m)])
                b = np.ones(m)
                # solve least squares
                sol, *_ = np.linalg.lstsq(Aeq, b, rcond=None)
                W[k * n + i] = sol
        # verify
        if _check(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-7):
            # rationalize
            Ur = _rationalize(U)
            Vr = _rationalize(V)
            Wr = _rationalize(W)
            if _check(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return {"U": Ur, "V": Vr, "W": Wr}
    return None


def construct():
    n, m, p = DIMENSIONS
    school = _schoolbook(n, m, p)
    best = school
    best_rank = n * m * p

    # Lower bound: max(n*m, m*p, p*n) = max(12,32,24)=32; also rank >= max flattening rank.
    # Try ranks from a modest value up to schoolbook-1.
    lb = max(n * m, m * p, p * n)
    # Search a window of ranks below schoolbook, with limited iterations each.
    seed = 12345
    for r in range(lb, best_rank):
        if _elapsed() > _DEADLINE - 10.0:
            break
        if r >= best_rank:
            break
        # more iterations for smaller r
        iters = 40 if r < 60 else 15
        res = _try_rank(r, n, m, p, iters, seed)
        seed += 1
        if res is not None:
            best = res
            best_rank = r
            break  # found a valid one; keep going to try smaller? we go upward, so this is the min found
            # since we iterate r upward, first success is minimal among searched
            break

    # ensure rational lists
    return best


# EVOLVE-BLOCK-END
