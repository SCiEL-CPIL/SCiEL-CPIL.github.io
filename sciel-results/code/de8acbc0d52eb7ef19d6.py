"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
Uses block Strassen (rank 22) as baseline, then searches for rank-21 via
BFGS optimization of the bilinear verification error.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0  # seconds, leaves 10s margin before 150s kill


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Full bilinear verification: for all (ia, ja, jb, kb, i, k),
    sum_r W[k*n+i, r] * U[ia*m+ja, r] * V[jb*p+kb, r] == delta(ia,i)*delta(ja,jb)*delta(kb,k)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]

    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = np.dot(W[Widx, :], U[Uidx, :] * V[Vidx, :])
                            if abs(actual - expected) > tol:
                                return False
    return True


def _verify_rational(U, V, W, n, m, p):
    """Exact rational verification using fractions.
    Returns True iff the tensor contraction is exactly the identity.
    """
    from fractions import Fraction
    
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    r = U.shape[1]
    
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1 if (ia == i and ja == jb and kb == k) else 0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = Fraction(0)
                            for rr in range(r):
                                u_val = Fraction(U[Uidx, rr])
                                v_val = Fraction(V[Vidx, rr])
                                w_val = Fraction(W[Widx, rr])
                                actual += u_val * v_val * w_val
                            if actual != expected:
                                return False
    return True


def _bilinear_error(U, V, W, n, m, p):
    """Compute the squared error of the bilinear verification.
    Error = sum over all (ia,ja,jb,kb,i,k) of (actual - expected)^2
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    total_err = 0.0
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = np.dot(W[Widx, :], U[Uidx, :] * V[Vidx, :])
                            total_err += (actual - expected) ** 2
    return total_err


def _try_ilp_rank_reduction(n, m, p, target_rank=21):
    """Use ILP-style search to find rank reduction from rank-22.
    
    Formulates the problem as: which subset of columns from the rank-22
    decomposition can be retained (or merged) to achieve target_rank while
    maintaining exact bilinear verification?
    
    Strategy:
    1. Build constraint matrix from rank-22 decomposition
    2. Use branch-and-bound to search for valid column subsets
    3. Verify candidates with exact rational arithmetic
    """
    from fractions import Fraction
    import itertools
    
    nm, mp, pn = n * m, m * p, p * n
    r0 = 22
    
    # Get rank-22 decomposition as Fractions
    U22, V22, W22 = _rank22_list(n, m, p)
    
    def to_frac_matrix(mat):
        return np.array([[Fraction(x) for x in row] for row in mat], dtype=object)
    
    U_frac = to_frac_matrix(U22)
    V_frac = to_frac_matrix(V22)
    W_frac = to_frac_matrix(W22)
    
    # Build the tensor constraint matrix
    # For each (ia, ja, jb, kb, i, k), we need:
    # sum_r U[ia*m+ja, r] * V[jb*p+kb, r] * W[k*n+i, r] = delta(ia,i)*delta(ja,jb)*delta(kb,k)
    
    # Precompute all constraint indices
    constraints = []
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1 if (ia == i and ja == jb and kb == k) else 0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            constraints.append((Uidx, Vidx, Widx, expected))
    
    def check_subset_valid(cols):
        """Check if keeping only these columns gives exact decomposition."""
        for Uidx, Vidx, Widx, expected in constraints:
            actual = Fraction(0)
            for r in cols:
                actual += U_frac[Uidx, r] * V_frac[Vidx, r] * W_frac[Widx, r]
            if actual != expected:
                return False
        return True
    
    def try_merge(cols_keep, col_merge_from, col_merge_to, c):
        """Try merging col_merge_from into col_merge_to with coefficient c."""
        # New column values
        U_new = {}
        V_new = {}
        W_new = {}
        
        for r in cols_keep:
            if r == col_merge_from:
                continue
            if r == col_merge_to:
                for Uidx in range(nm):
                    U_new[Uidx] = U_frac[Uidx, col_merge_to] + c * U_frac[Uidx, col_merge_from]
                for Vidx in range(mp):
                    V_new[Vidx] = V_frac[Vidx, col_merge_to] - c * V_frac[Vidx, col_merge_from]
                for Widx in range(pn):
                    W_new[Widx] = W_frac[Widx, col_merge_to]
            else:
                for Uidx in range(nm):
                    U_new[Uidx] = U_frac[Uidx, r]
                for Vidx in range(mp):
                    V_new[Vidx] = V_frac[Vidx, r]
                for Widx in range(pn):
                    W_new[Widx] = W_frac[Widx, r]
        
        # Check if merged decomposition is valid
        for Uidx, Vidx, Widx, expected in constraints:
            actual = Fraction(0)
            for r in cols_keep:
                if r == col_merge_from:
                    continue
                u_val = U_new.get(Uidx, Fraction(0))
                v_val = V_new.get(Vidx, Fraction(0))
                w_val = W_new.get(Widx, Fraction(0))
                actual += u_val * v_val * w_val
            if actual != expected:
                return None
        
        # Build result matrices
        result_cols = [r for r in cols_keep if r != col_merge_from]
        U_res = [[0]*len(result_cols) for _ in range(nm)]
        V_res = [[0]*len(result_cols) for _ in range(mp)]
        W_res = [[0]*len(result_cols) for _ in range(pn)]
        
        for new_idx, old_idx in enumerate(result_cols):
            for Uidx in range(nm):
                U_res[Uidx][new_idx] = float(U_new.get(Uidx, Fraction(0)))
            for Vidx in range(mp):
                V_res[Vidx][new_idx] = float(V_new.get(Vidx, Fraction(0)))
            for Widx in range(pn):
                W_res[Widx][new_idx] = float(W_new.get(Widx, Fraction(0)))
        
        return U_res, V_res, W_res
    
    # Search for valid rank-21 configurations
    all_cols = list(range(r0))
    
    # First: try simple column drops (rank 21 by removing one column)
    for drop in all_cols:
        if _remaining() < 15.0:
            break
        cols_keep = [c for c in all_cols if c != drop]
        if check_subset_valid(cols_keep):
            U_res = [[float(U_frac[i, c]) for c in cols_keep] for i in range(nm)]
            V_res = [[float(V_frac[i, c]) for c in cols_keep] for i in range(mp)]
            W_res = [[float(W_frac[i, c]) for c in cols_keep] for i in range(pn)]
            return U_res, V_res, W_res, target_rank
    
    # Second: try merging pairs with small coefficients
    c_options = [Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)]
    
    for drop_from in all_cols:
        if _remaining() < 10.0:
            break
        for merge_to in all_cols:
            if merge_to == drop_from:
                continue
            for c in c_options:
                if _remaining() < 5.0:
                    break
                cols_keep = all_cols  # Keep all, but merge drop_from into merge_to
                result = try_merge(cols_keep, drop_from, merge_to, c)
                if result is not None:
                    return result + (target_rank,)
    
    return None


def _rank22_list(n, m, p):
    """Construct rank-22 decomposition using block Strassen."""
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A: 0=A00, 1=A01, 3=A10, 4=A11
    # B: 0=B00, 1=B01, 4=B10, 5=B11
    # W: 0=C00, 2=C01, 1=C10, 3=C11

    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B: 2=B02, 3=B03, 6=B12, 7=B13
    # W: 4=C02, 6=C03, 5=C12, 7=C13

    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1

    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A: 2=A02, 5=A12
    # B: 8=B20, 9=B21, 10=B22, 11=B23
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # Start with rank-22
    best_U, best_V, best_W = _rank22_list(n, m, p)
    best_rank = 22

    # Verify
    if not _verify_full(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p

    if _remaining() < 30.0:
        return _to_dict(best_U, best_V, best_W)

    # Try to find rank-21 by optimizing from a perturbed rank-22
    # Strategy: take rank-22, zero out one column (or merge two columns),
    # then optimize the remaining parameters to minimize bilinear error.

    # Better approach: directly optimize rank-21 parameters from random init
    # using the bilinear error as objective.

    def _try_rank21():
        r = 21
        # Random initialization near rank-22 structure
        # Take rank-22 and randomly drop one column, then optimize
        U22, V22, W22 = _rank22_list(n, m, p)
        U22 = np.array(U22, dtype=float)
        V22 = np.array(V22, dtype=float)
        W22 = np.array(W22, dtype=float)

        best_err = float('inf')
        best_params = None

        for drop_col in range(22):
            if _remaining() < 25.0:
                break
            # Drop column drop_col
            keep = [c for c in range(22) if c != drop_col]
            U_init = U22[:, keep]
            V_init = V22[:, keep]
            W_init = W22[:, keep]

            # Flatten for optimization
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def objective(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error(U, V, W, n, m, p)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 200, 'gtol': 1e-6})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                continue

        if best_params is not None and best_err < 1e-8:
            x = best_params
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            if _verify_full(U, V, W, n, m, p, tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist(), r
        return None

    result21 = _try_rank21()
    if result21 is not None:
        U21, V21, W21, r21 = result21
        return _to_dict(U21, V21, W21)
    
    # Try ILP-based rank reduction search
    if _remaining() > 25.0:
        ilp_result = _try_ilp_rank_reduction(n, m, p, target_rank=21)
        if ilp_result is not None:
            U_ilp, V_ilp, W_ilp, r_ilp = ilp_result
            return _to_dict(U_ilp, V_ilp, W_ilp)

    # If rank-21 not found, try a few random restarts for rank-21
    if _remaining() > 30.0:
        def _random_rank21_attempt():
            r = 21
            x0 = np.random.randn(nm * r + mp * r + pn * r) * 0.1

            def objective(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error(U, V, W, n, m, p)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 300, 'gtol': 1e-7})
                if res.fun < 1e-10:
                    x = res.x
                    U = x[:nm * r].reshape(nm, r)
                    V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                    W = x[nm * r + mp * r:].reshape(pn, r)
                    if _verify_full(U, V, W, n, m, p, tol=1e-6):
                        return U.tolist(), V.tolist(), W.tolist(), r
            except Exception:
                pass
            return None

        for _ in range(3):
            if _remaining() < 20.0:
                break
            res = _random_rank21_attempt()
            if res is not None:
                U21, V21, W21, r21 = res
                return _to_dict(U21, V21, W21)

    return _to_dict(best_U, best_V, best_W)
