"""Matrix multiplication 3x3 x 3x4: rank-minimal exact tensor decomposition."""
import numpy as np
from scipy.optimize import least_squares
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 4
    # Tensor T[i, j, k] = 1 for the multiplication tensor (A[i,j]*B[j,k] -> C[i,k])
    # We seek rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where W follows trace convention W[k*n + i, r]

    # Build the multiplication tensor as a 3D array T[i, j, k] = 1 (it's all ones for the identity mapping)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in the standard basis.
    # The tensor we need to decompose: for each (i,j,k), the coefficient is 1.
    # So T is an n x m x p tensor of all 1s.
    T = np.ones((n, m, p), dtype=np.float64)

    # Reshape to matrix form for rank-1 decomposition
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # The constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1

    best_rank = n * m * p  # schoolbook fallback = 36
    best_result = None

    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Try to find lower rank via optimization
    # Known bounds: for 3x3 x 3x3, Strassen gives rank 23.
    # For 3x3 x 3x4, the rank is between 23 and 36.
    # Let's try ranks from low to high.

    # Precompute index mappings
    # For a given rank r, we have variables:
    # U: n*m*r, V: m*p*r, W: p*n*r
    # Total variables: r*(n*m + m*p + p*n) = r*(9 + 12 + 12) = 33*r

    # The equations: for each (i,j,k) in [0,n)x[0,m)x[0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Number of equations: n*m*p = 36

    def try_rank(r, num_restarts=20, max_iter=2000):
        """Try to find a rank-r decomposition."""
        t_start = time.time()
        nm = n * m
        mp = m * p
        pn = p * n
        num_eq = n * m * p  # 36

        # Precompute which (i,j,k) maps to which indices
        # For equation (i,j,k): uses U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        eq_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_indices.append((i * m + j, j * p + k, k * n + i))

        for restart in range(num_restarts):
            if time.time() > DEADLINE:
                return None
            if time.time() - t_start > 10:  # per-attempt time limit
                continue

            # Random initialization
            rng = np.random.RandomState(restart * 1000 + r)
            U_init = rng.randn(nm, r) * 0.5
            V_init = rng.randn(mp, r) * 0.5
            W_init = rng.randn(pn, r) * 0.5

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def residual(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                res = np.zeros(num_eq)
                for idx, (ui, vi, wi) in enumerate(eq_indices):
                    res[idx] = np.dot(np.dot(U[ui, :], V[vi, :]), W[wi, :]) - 1.0
                return res

            try:
                result = least_squares(residual, x0, method='lm', max_nfev=max_iter)
            except Exception:
                continue

            if result.cost < 1e-12:
                U = result.x[:nm * r].reshape(nm, r)
                V = result.x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = result.x[nm * r + mp * r:].reshape(pn, r)
                return U, V, W, r

        return None

    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = max(9,12,12) = 12
    # But realistically, for 3x3x3x4, known results suggest rank around 23-28
    # Let's try a range
    for r in range(20, 37):
        if time.time() > DEADLINE:
            break

        result = try_rank(r, num_restarts=15, max_iter=1500)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
            best_result = (U, V, W, rank)
            break  # Found the lowest rank, stop

    # If no lower rank found, use schoolbook
    if best_result is None:
        U, V, W, rank = schoolbook()
        best_rank = rank
    else:
        U, V, W, rank = best_result

    # Convert to list format with integer/half-integer entries if possible
    # First, try to round to nearest integer or half-integer
    def to_exact(mat):
        """Try to convert float matrix to integer/half-integer entries."""
        # Scale by 2 to check for half-integers
        scaled = mat * 2
        rounded = np.round(scaled)
        if np.allclose(scaled, rounded, atol=1e-6):
            # Entries are half-integers
            return rounded / 2.0
        # Try integers
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int
        # Return as-is (shouldn't happen for exact solutions)
        return mat

    U_exact = to_exact(U)
    V_exact = to_exact(V)
    W_exact = to_exact(W)

    # Verify the decomposition is exact
    def verify(Uv, Vv, Wv, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank_val):
                        s += Uv[i * m + j, c] * Vv[j * p + k, c] * Wv[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_exact, V_exact, W_exact, rank):
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
        U_exact = U
        V_exact = V
        W_exact = W

    # Convert to list of lists
    U_list = [[float(U_exact[i, c]) for c in range(rank)] for i in range(n * m)]
    V_list = [[float(V_exact[i, c]) for c in range(rank)] for i in range(m * p)]
    W_list = [[float(W_exact[i, c]) for c in range(rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
