"""Rewritten program for the refined Cohn-Goncalves Laguerre construction.
Uses a tridiagonal (Jacobi) parameterization and a 2-parameter Calogero-Sutherland
inspired initialization, combined with multi-start L-BFGS-B in log-space."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    m = 12
    n = 1

    start = time.time()
    DEADLINE = 630.0
    MARGIN = 30.0

    def ok():
        return (time.time() - start) < (DEADLINE - MARGIN)

    # Laguerre basis coefficients: L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1) ** j) * math.comb(k + 1, j + 1) / math.factorial(j)
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c ** 2 * (np.arange(13) + 1))
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    def cost_with_grad(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros_like(log_roots)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c ** 2 * (np.arange(13) + 1))
        if norm_sq <= 1e-10:
            return 1e10, np.zeros_like(log_roots)
        cost = 1.0 / norm_sq
        w = np.arange(13) + 1
        grad = np.zeros(12)
        for i in range(12):
            other_roots = np.delete(roots, i)
            coeffs_other_desc = np.poly(other_roots)
            p_other = coeffs_other_desc[::-1]
            dp_dr_i = -p_other
            dc_dr_i = A_T_inv @ dp_dr_i
            dnorm_sq_dr_i = 2.0 * np.sum(c * w * dc_dr_i)
            dcost_dr_i = -dnorm_sq_dr_i / (norm_sq * norm_sq)
            grad[i] = dcost_dr_i * roots[i]
        return cost, grad

    def tridiag_to_roots(params):
        """params: 23 floats, first 12 are diag, next 11 are offdiag.
        Returns the 12 positive roots (eigenvalues of a symmetric tridiag matrix
        made positive by shifting) and a success flag."""
        d = params[:12].copy()
        e = params[12:23].copy()
        # Make tridiagonal matrix
        T = np.diag(d, 0) + np.diag(e, 1) + np.diag(e, -1)
        eigvals = np.linalg.eigvalsh(T)
        # Shift to make positive: shift by -min + small epsilon
        shift = -eigvals.min() + 1e-4
        roots = eigvals + shift
        if np.any(roots <= 0) or np.any(np.isnan(roots)):
            return None
        return roots

    def tridiag_cost(params):
        roots = tridiag_to_roots(params)
        if roots is None:
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    def tridiag_cost_grad(params):
        roots = tridiag_to_roots(params)
        if roots is None:
            return 1e10, np.zeros(23)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c ** 2 * (np.arange(13) + 1))
        if norm_sq <= 1e-10:
            return 1e10, np.zeros(23)
        cost = 1.0 / norm_sq
        w = np.arange(13) + 1
        grad = np.zeros(23)
        # Numerical gradient for tridiag params is expensive; use finite diff for offdiag
        # Actually use the analytic root gradient and back-propagate through eigenvectors.
        # For simplicity, use central differences with small step.
        h = 1e-6
        for i in range(23):
            params_plus = params.copy()
            params_plus[i] += h
            roots_plus = tridiag_to_roots(params_plus)
            if roots_plus is None:
                grad[i] = 0.0
                continue
            norm_sq_plus = compute_norm_sq(roots_plus)
            if norm_sq_plus <= 1e-10:
                grad[i] = 0.0
                continue
            cost_plus = 1.0 / norm_sq_plus
            grad[i] = (cost_plus - cost) / h
        return cost, grad

    def calogero_init(a, b):
        """Solve Calogero-Sutherland equilibrium: for each i,
        sum_{j!=i} 1/(x_i - x_j)^2 = a*x_i + b/x_i.
        Use Newton-Raphson in 12D."""
        # Start with geometric
        x = 2.0 * (1.8 ** np.arange(12))
        for it in range(200):
            F = np.zeros(12)
            J = np.zeros((12, 12))
            for i in range(12):
                # F_i = sum_{j!=i} 1/(x_i - x_j)^2 - a*x_i - b/x_i
                s = 0.0
                for j in range(12):
                    if j != i:
                        s += 1.0 / (x[i] - x[j]) ** 2
                F[i] = s - a * x[i] - b / x[i]
                for j in range(12):
                    if j == i:
                        # dF_i/dx_i = sum_{k!=i} 2/(x_i - x_k)^3 + a + b/x_i^2
                        s2 = 0.0
                        for k in range(12):
                            if k != i:
                                s2 += 2.0 / (x[i] - x[k]) ** 3
                        J[i, i] = s2 + a + b / (x[i] ** 2)
                    else:
                        J[i, j] = -2.0 / (x[i] - x[j]) ** 3
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            x_new = x + dx
            if np.any(x_new <= 0):
                break
            if np.max(np.abs(dx)) < 1e-12:
                x = x_new
                break
            x = x_new
        if np.any(x <= 0):
            return None
        return x

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Phase 1: Calogero-inspired multi-start
    calogero_params = [
        (1.0, 1.0), (1.0, 10.0), (10.0, 1.0), (10.0, 10.0),
        (0.5, 0.5), (0.5, 5.0), (5.0, 0.5), (5.0, 5.0),
        (2.0, 2.0), (2.0, 20.0), (20.0, 2.0), (20.0, 20.0),
    ]
    for a, b in calogero_params:
        if not ok():
            break
        x0 = calogero_init(a, b)
        if x0 is None:
            continue
        log_roots = np.log(x0)
        try:
            res = minimize(cost_with_grad, log_roots, jac=True, method='L-BFGS-B',
                           bounds=[(-4.0, 12.0)] * 12,
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 2: Geometric/arithmetic multi-start
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    initial_points = [np.log(seed_roots)]
    for a in [2.0, 3.0, 5.0, 8.0]:
        for q in [1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
            r = a * (q ** np.arange(12))
            if r.max() < 50000:
                initial_points.append(np.log(r))
    np.random.seed(42)
    for _ in range(15):
        r = np.exp(np.random.uniform(-1.0, 8.0, 12))
        initial_points.append(np.log(r))

    for init_lr in initial_points:
        if not ok():
            break
        try:
            res = minimize(cost_with_grad, init_lr, jac=True, method='L-BFGS-B',
                           bounds=[(-4.0, 12.0)] * 12,
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 3: Tridiagonal parameterization multi-start
    # Initial tridiag: diag = mean of roots, offdiag = spread
    base_roots = 4.0 * (2.0 ** np.arange(12))
    d0 = np.full(12, np.mean(base_roots))
    e0 = np.full(11, np.std(base_roots) * 0.5)
    tridiag_starts = [np.concatenate([d0, e0])]
    # Perturb
    np.random.seed(99)
    for _ in range(8):
        d = d0 * np.exp(np.random.normal(0, 0.3, 12))
        e = e0 * np.exp(np.random.normal(0, 0.3, 11))
        tridiag_starts.append(np.concatenate([d, e]))
    for tparams in tridiag_starts:
        if not ok():
            break
        try:
            res = minimize(tridiag_cost_grad, tparams, jac=True, method='L-BFGS-B',
                           options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_val:
                roots_t = tridiag_to_roots(res.x)
                if roots_t is not None:
                    best_val = res.fun
                    best_log_roots = np.log(roots_t)
                    best_roots = roots_t
        except Exception:
            pass

    # Phase 4: DE search in log-space
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    for seed in seeds:
        if not ok():
            break
        try:
            res = differential_evolution(cost_func, [(-4.0, 12.0)] * 12,
                                         seed=seed, maxiter=100, popsize=15,
                                         tol=1e-10, polish=True, init='sobol')
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 5: Final high-precision polish
    if best_log_roots is not None and ok():
        try:
            res = minimize(cost_with_grad, best_log_roots, jac=True, method='L-BFGS-B',
                           bounds=[(-6.0, 14.0)] * 12,
                           options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 6: One more DE polish around best
    if best_log_roots is not None and ok():
        try:
            # Tighten bounds around best
            lr_best = best_log_roots
            lb = np.maximum(lr_best - 2.0, -6.0)
            ub = np.minimum(lr_best + 2.0, 14.0)
            bounds_tight = list(zip(lb, ub))
            res = differential_evolution(cost_func, bounds_tight,
                                         seed=777, maxiter=80, popsize=12,
                                         tol=1e-12, polish=True)
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 7: Final polish
    if best_log_roots is not None and ok():
        try:
            res = minimize(cost_with_grad, best_log_roots, jac=True, method='L-BFGS-B',
                           options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)

    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
