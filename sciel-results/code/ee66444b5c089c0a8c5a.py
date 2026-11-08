"""
Optimized program for matrix multiplication 2x4x5.
Strategy: Build a rank-36 Strassen decomposition, then aggressively search for
a rank-35 solution using:
  1. Drop-and-optimize with noise (multiple attempts)
  2. Drop-and-optimize from different starting points
  3. Random restarts with structured initialization
  4. Final high-precision polish with tighter tolerances
  5. Exact integer/half-integer rounding verification
"""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin before 150s kill

    # ── Build rank-36 Strassen-based decomposition ──────────────────────────
    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i

        col = 0

        def strassen_block(y, z, w):
            nonlocal col
            y00, y01, y10, y11 = y
            z00, z01, z10, z11 = z
            w0, w1, w2, w3 = w  # offsets for C[0,0], C[1,0], C[0,1], C[1,1]

            # M1 = (y00+y11)(z00+z11)
            U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
            V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
            W[w_idx(*w0)][col] += 1; W[w_idx(*w3)][col] += 1
            col += 1
            # M2 = (y10+y11)z00
            U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
            V[v_idx(*z00)][col] += 1
            W[w_idx(*w2)][col] += 1; W[w_idx(*w3)][col] -= 1
            col += 1
            # M3 = y00(z01-z11)
            U[u_idx(*y00)][col] += 1
            V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] -= 1
            W[w_idx(*w1)][col] += 1; W[w_idx(*w3)][col] += 1
            col += 1
            # M4 = y11(z10-z00)
            U[u_idx(*y11)][col] += 1
            V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] -= 1
            W[w_idx(*w0)][col] += 1; W[w_idx(*w2)][col] += 1
            col += 1
            # M5 = (y00+y01)z11
            U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
            V[v_idx(*z11)][col] += 1
            W[w_idx(*w0)][col] -= 1; W[w_idx(*w1)][col] += 1
            col += 1
            # M6 = (y10-y00)(z00+z01)
            U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] -= 1
            V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
            W[w_idx(*w3)][col] += 1
            col += 1
            # M7 = (y01-y11)(z10+z11)
            U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] -= 1
            V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
            W[w_idx(*w0)][col] += 1
            col += 1

        # Four 2x2x2 Strassen blocks
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)],
                       [(2,0),(3,0),(2,1),(3,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)],
                       [(2,0),(3,0),(2,1),(3,1)])

        # 2x2 x 2x1 blocks (4 terms each)
        for row in range(2):
            for col_a in range(2):
                U[u_idx(row, col_a)][col] += 1
                V[v_idx(col_a, 4)][col] += 1
                W[w_idx(4, row)][col] += 1
                col += 1
        for row in range(2):
            for col_a in range(2, 4):
                U[u_idx(row, col_a)][col] += 1
                V[v_idx(col_a, 4)][col] += 1
                W[w_idx(4, row)][col] += 1
                col += 1

        assert col == rank
        return {"U": U, "V": V, "W": W}

    best_result = build_strassen_36()
    best_rank = 36

    # ── Verification ─────────────────────────────────────────────────────────
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check all target entries equal 1
        for a, b, c in target_indices:
            val = np.dot(W[c, :], U[a, :] * V[b, :])
            if abs(val - 1.0) > 1e-8:
                return False
        # Check ALL non-target entries are 0 (exhaustive check)
        target_set = set(target_indices)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target_set:
                        val = np.dot(W[c, :], U[a, :] * V[b, :])
                        if abs(val) > 1e-8:
                            return False
        return True

    # ── Precompute indices ───────────────────────────────────────────────────
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))

    # Build check set: near-miss (differ in exactly one of the three indices)
    target_set = set(target_indices)
    check_indices = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) in target_set:
                    continue
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                matches = (j == j2) + (k == k2) + (i == i2)
                if matches >= 2:
                    check_indices.append((a, b, c))

    # Add some far indices
    rng_init = np.random.RandomState(99)
    all_non_target = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)
                      if (a, b, c) not in target_set]
    for _ in range(200):
        check_indices.append(rng_init.choice(len(all_non_target), replace=False) if len(all_non_target) > 0 else (0,0,0))
    check_indices = [all_non_target[i] for i in rng_init.choice(len(all_non_target), size=min(200, len(all_non_target)), replace=False)]
    check_indices = list(set(check_indices))

    # ── Objective function (vectorized) ─────────────────────────────────────
    target_A = np.array([x[0] for x in target_indices])
    target_B = np.array([x[1] for x in target_indices])
    target_C = np.array([x[2] for x in target_indices])
    check_A = np.array([x[0] for x in check_indices]) if check_indices else np.array([])
    check_B = np.array([x[1] for x in check_indices]) if check_indices else np.array([])
    check_C = np.array([x[2] for x in check_indices]) if check_indices else np.array([])

    def objective(params, rank):
        num_u = n * m * rank
        num_v = m * p * rank
        U = params[:num_u].reshape((n * m, rank))
        V = params[num_u:num_u + num_v].reshape((m * p, rank))
        W = params[num_u + num_v:]

        # Target entries should equal 1
        # Vectorized: for each target (a,b,c), compute sum_k U[a,k]*V[b,k]*W[c,k]
        err = 0.0
        for idx in range(len(target_A)):
            a, b, c = target_A[idx], target_B[idx], target_C[idx]
            val = np.dot(W[c, :], U[a, :] * V[b, :])
            err += (val - 1.0) ** 2

        # Check entries should be 0
        for idx in range(len(check_A)):
            a, b, c = check_A[idx], check_B[idx], check_C[idx]
            val = np.dot(W[c, :], U[a, :] * V[b, :])
            err += val ** 2

        return err

    def objective_with_grad(params, rank):
        """Objective with analytical gradient for faster convergence."""
        num_u = n * m * rank
        num_v = m * p * rank
        U = params[:num_u].reshape((n * m, rank))
        V = params[num_u:num_u + num_v].reshape((m * p, rank))
        W = params[num_u + num_v:].reshape((p * n, rank))

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        err = 0.0

        # Target entries
        for idx in range(len(target_A)):
            a, b, c = target_A[idx], target_B[idx], target_C[idx]
            uv = U[a, :] * V[b, :]
            val = np.dot(W[c, :], uv)
            residual = val - 1.0
            err += residual ** 2
            # Gradient contributions
            grad_U[a, :] += 2 * residual * V[b, :] * W[c, :]
            grad_V[b, :] += 2 * residual * U[a, :] * W[c, :]
            grad_W[c, :] += 2 * residual * uv

        # Check entries
        for idx in range(len(check_A)):
            a, b, c = check_A[idx], check_B[idx], check_C[idx]
            uv = U[a, :] * V[b, :]
            val = np.dot(W[c, :], uv)
            err += val ** 2
            grad_U[a, :] += 2 * val * V[b, :] * W[c, :]
            grad_V[b, :] += 2 * val * U[a, :] * W[c, :]
            grad_W[c, :] += 2 * val * uv

        return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # ── Optimization strategies ─────────────────────────────────────────────
    U0 = np.array(best_result['U'], dtype=float)
    V0 = np.array(best_result['V'], dtype=float)
    W0 = np.array(best_result['W'], dtype=float)

    def try_solve(U_init, V_init, W_init, rank, max_iter=300, ftol=1e-14, gtol=1e-10, use_grad=True):
        """Run L-BFGS-B and return rounded solution or None."""
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        try:
            if use_grad:
                res = minimize(objective_with_grad, params0, args=(rank,), method='L-BFGS-B',
                               jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
            else:
                res = minimize(objective, params0, args=(rank,), method='L-BFGS-B',
                               options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
            if res.fun < 1e-6:
                num_u = n * m * rank
                num_v = m * p * rank
                U_sol = res.x[:num_u].reshape((n * m, rank))
                V_sol = res.x[num_u:num_u + num_v].reshape((m * p, rank))
                W_sol = res.x[num_u + num_v:].reshape((p * n, rank))

                # Try rounding to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r

                # Try rounding to integers
                U_r = np.round(U_sol)
                V_r = np.round(V_sol)
                W_r = np.round(W_sol)
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r

                # Try without rounding (check if already close)
                if verify(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        return None

    rng = np.random.RandomState(42)

    # Strategy 1: Drop one column from rank-36, add noise, optimize for rank-35
    for attempt in range(20):
        if time.time() > deadline:
            break
        drop_idx = rng.randint(0, 35)
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()

        noise = rng.uniform(0.01, 0.15)
        U_init += rng.normal(0, noise, U_init.shape)
        V_init += rng.normal(0, noise, V_init.shape)
        W_init += rng.normal(0, noise, W_init.shape)

        result = try_solve(U_init, V_init, W_init, 35, max_iter=400, ftol=1e-14, gtol=1e-10, use_grad=True)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Strategy 2: Drop one column, no noise (pure projection)
    for drop_idx in range(36):
        if time.time() > deadline:
            break
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()

        result = try_solve(U_init, V_init, W_init, 35, max_iter=500, ftol=1e-15, gtol=1e-11, use_grad=True)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Strategy 3: Random structured initialization for rank-35
    for attempt in range(15):
        if time.time() > deadline:
            break
        # Start from rank-36, zero out a random column, then random restart
        U_init = U0.copy()
        V_init = V0.copy()
        W_init = W0.copy()
        zero_col = rng.randint(0, 35)
        U_init[:, zero_col] = 0
        V_init[:, zero_col] = 0
        W_init[:, zero_col] = 0

        # Keep only 35 columns (drop the zeroed one)
        keep = [i for i in range(36) if i != zero_col]
        U_init = U_init[:, keep].copy()
        V_init = V_init[:, keep].copy()
        W_init = W_init[:, keep].copy()

        # Add small perturbation to the last column
        U_init[:, -1] += rng.normal(0, 0.3, n*m)
        V_init[:, -1] += rng.normal(0, 0.3, m*p)
        W_init[:, -1] += rng.normal(0, 0.3, p*n)

        result = try_solve(U_init, V_init, W_init, 35, max_iter=350, ftol=1e-13, gtol=1e-9, use_grad=True)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Strategy 4: Try rank-34 as well (aggressive)
    for attempt in range(10):
        if time.time() > deadline:
            break
        drop_idx = rng.randint(0, 35)
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()
        # Drop another column
        drop_idx2 = rng.randint(0, 34)
        U_init = np.delete(U_init, drop_idx2, axis=1).copy()
        V_init = np.delete(V_init, drop_idx2, axis=1).copy()
        W_init = np.delete(W_init, drop_idx2, axis=1).copy()

        noise = rng.uniform(0.05, 0.2)
        U_init += rng.normal(0, noise, U_init.shape)
        V_init += rng.normal(0, noise, V_init.shape)
        W_init += rng.normal(0, noise, W_init.shape)

        result = try_solve(U_init, V_init, W_init, 34, max_iter=500, ftol=1e-14, gtol=1e-10, use_grad=True)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return best_result
