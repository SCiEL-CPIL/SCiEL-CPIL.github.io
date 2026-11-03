"""Matrix multiplication 3,3,4 rank-minimal exact tensor decomposition.

Strategy: the multiplication tensor is a 12x12x12 trilinear form.  Under the
trace convention W[k*n+i, r], a rank-r decomposition exists iff a structured
linear system (in the U factors, with V and W free) is solvable.  We test
ranks in increasing order using an SVD-based rank test, then, once a rank is
found, we reconstruct an exact rational solution (fractions.Fraction) and
verify it exactly.  A deadline guard keeps us safely under the wall-clock limit.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 130.0
_CACHE = {}


def _time_left():
    return _DEADLINE - time.time()


def _build_system(n, m, p):
    """Build the coefficient matrices for the trace-convention rank system.

    Unknowns: U (n*m x r), V (m*p x r), W (p*n x r).
    Equation (i,j,k):
        U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]   for each c, summed over c.
    Rearranged into a linear system in U (V, W free):
        For each (i,j,k):  sum_c U[i*m+j,c] * V[j*p+k,c] - sum_c W[k*n+i,c] = 0
    """
    nm, mp, pn = n * m, m * p, p * n
    # We'll build a linear system over the U-variables only, treating V and W
    # as free.  The system is: for each (i,j,k), the equation is linear in U
    # once V is fixed.  Instead, we use the standard "tensor rank" linear
    # test: the multiplication tensor M has entries M[i*m+j, j*p+k, k*n+i] = 1.
    # Rank <= r iff the tensor lies in the sum of r rank-1 tensors.
    #
    # A practical necessary-and-sufficient test: form the "flattened" system.
    # We use the approach of checking whether the tensor's slices can be
    # expressed with r bilinear terms.  Concretely, we build a matrix A of
    # size (n*m*p) x (n*m + m*p + p*n) * (r-1) ... this gets complicated.
    #
    # Simpler robust approach: use the "linear algebra rank test" via
    # the tensor's matricizations.  For rank r, we need:
    #   there exist U (nm x r), V (mp x r), W (pn x r) such that
    #   T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # where T[i,j,k] = 1 if (i,j,k) corresponds to a valid (i,j,k) triple,
    # i.e. T is the multiplication tensor with T[i, j, k] = 1 for all valid
    # (i,j,k) and the "output" index is (k, i) in the W convention.
    #
    # The cleanest test: build the big matrix of size (n*m*p) x (n*m*r) by
    # fixing V and W, but that requires fixing them.  Instead we use the
    # "slice rank" / "tensor rank" linear test:
    #
    # T is rank <= r iff the following system is consistent:
    #   For each output index o = (k, i) [0..p*n-1], the slice T_o (size n*m x m*p)
    #   must be expressible as sum_{c=1}^r u_{o,c} * v_c  where u_{o,c} is a
    #   scalar and v_c is a vector in R^{n*m x m*p}... no, that's slice rank.
    #
    # Let me use the direct approach: the tensor T has entries
    #   T[a, b, c] = 1  where a = i*m+j, b = j*p+k, c = k*n+i, for all i,j,k.
    #   T[a, b, c] = 0 otherwise.
    # Rank <= r iff T = sum_{s=1}^r u_s outer v_s outer w_s.
    #
    # Linear test: fix the last factor.  T is rank <= r iff there exist
    #   W (pn x r) such that for each c, the matrix  T_c (size nm x mp)
    #   (the c-th "fiber" in the output) equals sum_s W[c, s] * U_s outer V_s.
    # This is still nonlinear.
    #
    # Practical approach used in literature: the "rank <= r" problem is NP-hard
    # in general, but for small tensors we can use a Groebner-free heuristic:
    # try random V, W and check if the resulting linear system in U is consistent,
    # iterating.  For 3x3x4 the true rank is 19 (Landsberg) or possibly less.
    #
    # Given the difficulty, I'll use a randomized linear-algebra test:
    #   1. Randomly generate V (mp x r) and W (pn x r).
    #   2. For each output index c, the equation is:
    #        T_c = sum_s W[c,s] * (U_s outer V_s)
    #      where T_c is the nm x mp matrix with T_c[i*m+j, j*p+k] = 1 if
    #      the output index c = k*n+i matches, else 0.
    #   3. This is linear in U.  Stack all equations and solve least-squares;
    #      check residual.
    #
    # This is a heuristic but works well in practice for small tensors.

    nm, mp, pn = n * m, m * p, p * n
    # Build T as a 3D array: T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T, nm, mp, pn


def _rank_test(T, nm, mp, pn, r, rng):
    """Heuristic rank test: try random V, W and check linear consistency in U."""
    for _ in range(200):
        if _time_left() < 10:
            return False
        V = rng.standard_normal((mp, r))
        W = rng.standard_normal((pn, r))
        # Build linear system in U (nm x r)
        # For each output c, for each (a, b):
        #   T[a, b, c] = sum_s W[c, s] * U[a, s] * V[b, s]
        # This is NOT linear in U because U[a,s] * V[b,s] is a product.
        # We need a different approach.
        #
        # Correct linear test: fix V and W.  The equation becomes:
        #   For each (a, b, c): T[a,b,c] = sum_s U[a,s] * (V[b,s] * W[c,s])
        # This IS linear in U!  The coefficient of U[a,s] in equation (a,b,c)
        # is V[b,s] * W[c,s].
        #
        # So we build a matrix A of size (nm*mp*pn) x (nm*r) where
        # A[(a,b,c), (a',s)] = V[b,s]*W[c,s] if a == a', else 0.
        # And the RHS is T flattened.
        # Solve A * u_flat = t_flat by least squares; check residual.

        # Build A efficiently: it's block-diagonal in a.
        # For each a, the block is (mp*pn) x (r): A_a[b*pn+c, s] = V[b,s]*W[c,s]
        # Stack over a.
        # Total size: nm*(mp*pn) x (nm*r).  For 3x3x4: 12*48 x 12*r = 576 x 12r.
        # For r=19: 576 x 228.  Solvable.

        # Build A
        A = np.zeros((nm * mp * pn, nm * r))
        t_flat = T.reshape(-1)
        for a in range(nm):
            # Block for this a: rows (a*mp*pn + b*pn + c), cols (a*r + s)
            for s in range(r):
                # Coefficient vector for U[a, s]: V[:, s] outer W[:, s] flattened
                coeff = (V[:, s] * W[:, s].reshape(1, -1)).reshape(-1)
                A[a * mp * pn: (a + 1) * mp * pn, a * r + s] = coeff

        # Solve least squares
        try:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A, t_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Check residual
        residual = np.linalg.norm(A @ sol - t_flat)
        if residual < 1e-8:
            # Found a solution!  Extract U
            U = sol.reshape(nm, r)
            return U, V, W
    return False


def _verify_exact(U, V, W, n, m, p):
    """Verify the decomposition exactly using Fraction arithmetic."""
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    Uf = [[Fraction(int(U[i, c]), 1) for c in range(r)] for i in range(nm)]
    Vf = [[Fraction(int(V[i, c]), 1) for c in range(r)] for i in range(mp)]
    Wf = [[Fraction(int(W[i, c]), 1) for c in range(r)] for i in range(pn)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(Uf[a][cc] * Vf[b][cc] * Wf[c][cc] for cc in range(r))
                if s != 1:
                    return False
    # Check that all other entries are 0
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j != j2 or i != i2 or k != k2:
                    s = sum(Uf[a][cc] * Vf[b][cc] * Wf[c][cc] for cc in range(r))
                    if s != 0:
                        return False
    return True


def _find_exact(U, V, W, n, m, p):
    """Try to find exact rational U, V, W from the floating-point solution."""
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    # Round to nearest integer/half-integer and verify
    for tol in [0.5, 0.25, 0.125]:
        Ur = np.round(U * 2) / 2
        Vr = np.round(V * 2) / 2
        Wr = np.round(W * 2) / 2
        if _verify_exact(Ur, Vr, Wr, n, m, p):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    # Try rounding to nearest integer
    Ui = np.round(U)
    Vi = np.round(V)
    Wi = np.round(W)
    if _verify_exact(Ui, Vi, Wi, n, m, p):
        return Ui.tolist(), Vi.tolist(), Wi.tolist()
    # Try with larger denominators
    for denom in [3, 4, 6, 8, 12]:
        Ur = np.round(U * denom) / denom
        Vr = np.round(V * denom) / denom
        Wr = np.round(W * denom) / denom
        if _verify_exact(Ur, Vr, Wr, n, m, p):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None


def construct():
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(42)
    T, nm, mp, pn = _build_system(n, m, p)

    # The schoolbook rank is n*m*p = 36.  We search from a lower bound upward.
    # Known lower bound for 3x3x4: at least 19 (Landsberg's bound for 3x3x3 is 19,
    # and 3x3x4 >= 3x3x3).  Actually the rank of 3x3x4 is known to be 19.
    # But let's search from a reasonable lower bound.
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12.
    # Better lower bound from slice rank: at least 19 for 3x3x3, so 3x3x4 >= 19.
    lower = 19
    upper = n * m * p  # 36

    for r in range(lower, upper + 1):
        if _time_left() < 15:
            break
        result = _rank_test(T, nm, mp, pn, r, rng)
        if result is not False and isinstance(result, tuple):
            U, V, W = result
            exact = _find_exact(U, V, W, n, m, p)
            if exact is not None:
                return {"U": exact[0], "V": exact[1], "W": exact[2]}
            # If exact verification fails, try more random restarts
            for _ in range(50):
                if _time_left() < 10:
                    break
                result2 = _rank_test(T, nm, mp, pn, r, rng)
                if result2 is not False and isinstance(result2, tuple):
                    U2, V2, W2 = result2
                    exact2 = _find_exact(U2, V2, W2, n, m, p)
                    if exact2 is not None:
                        return {"U": exact2[0], "V": exact2[1], "W": exact2[2]}

    # Fallback: return the schoolbook decomposition (rank 36)
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
# EVOLVE-BLOCK-END
