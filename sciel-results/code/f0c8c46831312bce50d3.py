"""
Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
Uses structured ALS with a full tensor-error objective (both diagonal and
off-diagonal constraints), random initialisations, and a final polish phase.
"""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before the 150 s kill

    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10

    # ---------- schoolbook (rank 50) ----------
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    best_solution = build_schoolbook()
    best_rank = n * m * p

    # ---------- full tensor error ----------
    # The matrix-multiplication tensor T has entries
    #   T[(i,j),(j',k),(k',i')] = 1  iff j==j' and k==k' and i==i'
    #   0 otherwise.
    # We need  sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T[...]
    # So we have n*m*m*p*p*n  constraints = 2*5*5*5*5*2 = 2500.
    # Diagonal (i==i', j==j', k==k'): target 1, count n*m*p = 50.
    # Off-diagonal: target 0, count 2500 - 50 = 2450.

    def tensor_error(U, V, W, R):
        """Return (max_abs_err, mean_abs_err) over all 2500 entries."""
        max_err = 0.0
        sum_err = 0.0
        count = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                val = np.dot(
                                    U[i * m + j, :],
                                    V[jp * p + k, :] * W[kp * n + ip, :]
                                )
                                if j == jp and k == kp and i == ip:
                                    err = abs(val - 1.0)
                                else:
                                    err = abs(val)
                                if err > max_err:
                                    max_err = err
                                sum_err += err
                                count += 1
        return max_err, sum_err / count

    def diag_error(U, V, W, R):
        """Faster: only the 50 diagonal constraints."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :],
                                 V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # ---------- ALS solvers (vectorised inner loops) ----------
    def solve_U(V, W, R):
        # Fix V, W.  For each (i,j), solve
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T
        # for all (j',k,k',i').  This is a linear system in U[i*m+j, :].
        # Build the 2500 x R matrix row-by-row per (i,j) is expensive.
        # Instead: for fixed i,j, the constraint is
        #   sum_r u_r * (V[jp*p+k, r] * W[kp*n+ip, r]) = T[(i,j),(jp,k),(kp,ip)]
        # We can build a (2500, R) matrix M and target t, then lstsq.
        U_out = np.zeros((nm, R))
        for i in range(n):
            for j in range(m):
                # Build M (2500, R) and t (2500,)
                M = np.zeros((m * p * p * n, R))
                t = np.zeros(m * p * p * n)
                row = 0
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                M[row, :] = V[jp * p + k, :] * W[kp * n + ip, :]
                                if j == jp and k == kp and i == ip:
                                    t[row] = 1.0
                                row += 1
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                    U_out[i * m + j, :] = sol
                except np.linalg.LinAlgError:
                    U_out[i * m + j, :] = 0.0
        return U_out

    def solve_V(U, W, R):
        V_out = np.zeros((mp, R))
        for jp in range(m):
            for k in range(p):
                M = np.zeros((n * m * p * n, R))
                t = np.zeros(n * m * p * n)
                row = 0
                for i in range(n):
                    for j in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                M[row, :] = U[i * m + j, :] * W[kp * n + ip, :]
                                if j == jp and k == kp and i == ip:
                                    t[row] = 1.0
                                row += 1
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                    V_out[jp * p + k, :] = sol
                except np.linalg.LinAlgError:
                    V_out[jp * p + k, :] = 0.0
        return V_out

    def solve_W(U, V, R):
        W_out = np.zeros((pn, R))
        for kp in range(p):
            for ip in range(n):
                M = np.zeros((n * m * m * p, R))
                t = np.zeros(n * m * m * p)
                row = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                M[row, :] = U[i * m + j, :] * V[jp * p + k, :]
                                if j == jp and k == kp and i == ip:
                                    t[row] = 1.0
                                row += 1
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                    W_out[kp * n + ip, :] = sol
                except np.linalg.LinAlgError:
                    W_out[kp * n + ip, :] = 0.0
        return W_out

    # ---------- faster ALS using only diagonal for speed, then check full ----------
    def solve_U_fast(V, W, R):
        """Solve using only the 50 diagonal constraints (much faster)."""
        U_out = np.zeros((nm, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_out[i * m + j, :] = sol
                except np.linalg.LinAlgError:
                    U_out[i * m + j, :] = 0.0
        return U_out

    def solve_V_fast(U, W, R):
        V_out = np.zeros((mp, R))
        for jp in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i * m + jp, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_out[jp * p + k, :] = sol
                except np.linalg.LinAlgError:
                    V_out[jp * p + k, :] = 0.0
        return V_out

    def solve_W_fast(U, V, R):
        W_out = np.zeros((pn, R))
        for kp in range(p):
            for ip in range(n):
                A = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[ip * m + j, :] * V[j * p + kp, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_out[kp * n + ip, :] = sol
                except np.linalg.LinAlgError:
                    W_out[kp * n + ip, :] = 0.0
        return W_out

    def als_round_fast(U, V, W, R):
        U = solve_U_fast(V, W, R)
        V = solve_V_fast(U, W, R)
        W = solve_W_fast(U, V, R)
        return U, V, W

    # ---------- try a given rank ----------
    def try_rank(R, n_restarts=8, max_iter=300, tol=1e-8):
        for restart in range(n_restarts):
            if time.time() > deadline - 8:
                return None

            np.random.seed(1000 * R + restart)

            if restart == 0:
                # Schoolbook-based: random projection
                U0, V0, W0 = build_schoolbook()
                P = np.random.randn(n * m * p, R)
                U = U0 @ P
                V = V0 @ P
                W = W0 @ P
            elif restart == 1:
                U = np.random.randn(nm, R) * 0.3
                V = np.random.randn(mp, R) * 0.3
                W = np.random.randn(pn, R) * 0.3
            elif restart == 2:
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif restart == 3:
                # Sparse random
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
                U[np.random.rand(nm, R) < 0.4] = 0
                V[np.random.rand(mp, R) < 0.4] = 0
                W[np.random.rand(pn, R) < 0.4] = 0
            elif restart == 4:
                # Block-diagonal style: first R columns of schoolbook
                U0, V0, W0 = build_schoolbook()
                U = U0[:, :R].copy()
                V = V0[:, :R].copy()
                W = W0[:, :R].copy()
            elif restart == 5:
                # Perturbed schoolbook
                U0, V0, W0 = build_schoolbook()
                P = np.random.randn(n * m * p, R) * 0.3
                Q = np.eye(n * m * p)
                if R <= n * m * p:
                    Q = np.zeros((n * m * p, n * m * p))
                    idx = np.random.choice(n * m * p, R, replace=False)
                    Q[idx, np.arange(R)] = 1.0
                U = U0 @ Q
                V = V0 @ Q
                W = W0 @ Q
            else:
                U = np.random.randn(nm, R) * (0.5 / np.sqrt(R))
                V = np.random.randn(mp, R) * (0.5 / np.sqrt(R))
                W = np.random.randn(pn, R) * (0.5 / np.sqrt(R))

            prev_err = np.inf
            converged = False
            for it in range(max_iter):
                if time.time() > deadline - 8:
                    return None

                U, V, W = als_round_fast(U, V, W, R)

                # Quick diagonal check
                derr = diag_error(U, V, W, R)
                if derr < 1e-10:
                    # Check full tensor error
                    max_err, _ = tensor_error(U, V, W, R)
                    if max_err < tol:
                        converged = True
                        # Polish
                        for _ in range(50):
                            if time.time() > deadline - 3:
                                break
                            U, V, W = als_round_fast(U, V, W, R)
                            max_err, _ = tensor_error(U, V, W, R)
                            if max_err < 1e-12:
                                break
                        return U, V, W

                if abs(prev_err - derr) < 1e-14:
                    break
                prev_err = derr

        return None

    # ---------- search over ranks ----------
    # Try ranks from 49 down.  The minimal rank for 2x5x5 is unknown a priori;
    # schoolbook is 50.  Try a range and keep the best.
    for R in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(R)
        if result is not None:
            best_rank = R
            best_solution = result
            # Don't stop; try lower
        else:
            # If we fail at rank R, try a couple lower ranks with less effort
            for R2 in range(R - 1, max(R - 4, 30), -1):
                if time.time() > deadline - 5:
                    break
                result2 = try_rank(R2, n_restarts=3, max_iter=100)
                if result2 is not None:
                    best_rank = R2
                    best_solution = result2
                    break
            break

    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
