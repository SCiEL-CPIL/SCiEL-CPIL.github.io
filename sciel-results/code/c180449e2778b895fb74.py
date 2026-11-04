"""Improved: efficient 12-DOF optimization with generalized Laguerre multi-start, Calogero-Sutherland equilibrium
initialization, and high-precision log-space polish. Uses analytic gradients for C4 and L-BFGS-B with diverse
initializations from L_12^(alpha) and Calogero-Sutherland stationary points.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """int poly(x) e^{-x} dx over [0,inf) for ascending coefficient order."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Build P(x)=prod(x-x_i)^2 and P2=P^2 in ascending coeff order."""
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2/I1^2 with optional gradient."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Gradient via prefix/suffix products
    n = len(roots)
    prefix = [None] * n
    prefix[0] = np.ones(1)
    for j in range(n - 1):
        prefix[j + 1] = polymul(prefix[j], np.array([roots[j]**2, -2.0*roots[j], 1.0]))

    suffix = [None] * n
    suffix[n - 1] = np.ones(1)
    for j in range(n - 2, -1, -1):
        suffix[j] = polymul(suffix[j + 1], np.array([roots[j]**2, -2.0*roots[j], 1.0]))

    grad = np.zeros(n)
    for i in range(n):
        if i == 0:
            Q_i = suffix[0].copy()
        elif i == n - 1:
            Q_i = prefix[n - 1].copy()
        else:
            Q_i = polymul(prefix[i], suffix[i])

        int_Qi = _poly_moment(Q_i)
        xQi = polymul(Q_i, np.array([0.0, 1.0]))
        int_xQi = _poly_moment(xQi)
        int_P_over_xi = int_xQi - roots[i] * int_Qi
        dI1_i = 2.0 * int_P_over_xi

        S_i = polymul(Q_i, Q_i)
        x3Si = polymul(S_i, np.array([0.0, 0.0, 0.0, 1.0]))
        x2Si = polymul(S_i, np.array([0.0, 0.0, 1.0]))
        xSi = polymul(S_i, np.array([0.0, 1.0]))
        int_x3Si = _poly_moment(x3Si)
        int_x2Si = _poly_moment(x2Si)
        int_xSi = _poly_moment(xSi)
        int_Si = _poly_moment(S_i)
        dI2_i = 4.0 * (int_x3Si - 3.0*roots[i]*int_x2Si + 3.0*roots[i]**2*int_xSi - roots[i]**3*int_Si)

        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)

    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _exp_objective_grad(log_roots):
    """Objective in log-space: x_i = exp(log_roots_i), removes positivity constraint."""
    x = np.exp(log_roots)
    c4, grad_x = _compute_c4(x, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(log_roots)
    grad_log = x * grad_x
    return c4, grad_log

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def _calogero_sutherland_roots(beta, n=12):
    """Solve for n positive roots as stationary points of Calogero-Sutherland model:
    sum_{j!=i} 1/(x_i - x_j)^2 + x_i - beta/x_i^2 = 0
    This gives a smooth family of root configurations parameterized by beta > 0.
    Uses Newton-Raphson with good initialization from Laguerre roots.
    """
    xg, _ = roots_laguerre(n)
    x = np.sort(xg).copy().astype(np.float64)
    
    def residual(x):
        r = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += 1.0 / (x[i] - x[j])**2
            r[i] = s + x[i] - beta / x[i]**2
        return r
    
    def jacobian(x):
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j != i:
                    J[i, j] = -2.0 / (x[i] - x[j])**3
            # d/dx_i of sum_{j!=i} 1/(x_i-x_j)^2 = sum_{j!=i} (-2)/(x_i-x_j)^3
            # d/dx_i of x_i = 1
            # d/dx_i of -beta/x_i^2 = 2*beta/x_i^3
            J[i, i] = sum(-2.0 / (x[i] - x[j])**3 for j in range(n) if j != i) + 1.0 + 2.0*beta/x[i]**3
        return J
    
    for iteration in range(100):
        r = residual(x)
        if np.max(np.abs(r)) < 1e-14:
            break
        J = jacobian(x)
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]
            except:
                break
        # Damped step
        alpha_step = 1.0
        for _ in range(30):
            x_new = x + alpha_step * dx
            if np.all(x_new > 0) and np.all(np.isfinite(x_new)):
                r_new = residual(x_new)
                if np.max(np.abs(r_new)) < np.max(np.abs(r)):
                    x = x_new
                    break
            alpha_step *= 0.5
        else:
            break
    
    x = np.sort(x)
    x = np.maximum(x, 1e-10)
    return x

def _rational_potential_roots(params, base_roots):
    """Solve for 12 positive roots as critical points of V(x) = sum_k a_k/(x-c_k) + x.
    params = [c1, c2, c3, a1, a2, a3, s1, s2, s3] (9 params)
    Stationarity: 2*sum_{j!=i} 1/(x_i - x_j) + V'(x_i) = 0
    where V'(x) = -sum_k a_k/(x-c_k)^2 + 1
    
    We use Newton-Raphson to solve the 12 coupled equations, initialized from base_roots.
    The parameters c_k (poles) and a_k (amplitudes) are optimized externally.
    For the inner solve, we treat c_k and a_k as fixed and solve for x_i.
    """
    n = 12
    c1, c2, c3 = params[0], params[1], params[2]
    a1, a2, a3 = params[3], params[4], params[5]
    s1, s2, s3 = params[6], params[7], params[8]
    
    # Poles must be outside the root interval [0, inf) for stability
    # We'll use poles at negative values or well-separated positive values
    # Actually, for the potential to create 12 critical points in (0,inf),
    # we need poles interleaved or outside. Let's use poles at negative positions
    # so V(x) is smooth on (0,inf).
    
    def V_prime(x):
        """V'(x) = -a1/(x-c1)^2 - a2/(x-c2)^2 - a3/(x-c3)^2 + 1"""
        result = np.ones_like(x)
        result -= a1 / (x - c1)**2
        result -= a2 / (x - c2)**2
        result -= a3 / (x - c3)**2
        return result
    
    def stationarity_residual(x):
        """r_i = 2*sum_{j!=i} 1/(x_i - x_j) + V'(x_i)"""
        r = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += 1.0 / (x[i] - x[j])
            r[i] = 2.0 * s + V_prime(x[i])
        return r
    
    # Newton-Raphson solve
    x = base_roots.copy().astype(np.float64)
    for iteration in range(50):
        r = stationarity_residual(x)
        if np.max(np.abs(r)) < 1e-12:
            break
        
        # Jacobian: dr_i/dx_j = -2/(x_i-x_j)^2 for i!=j, dr_i/dx_i = 2*sum_{j!=i} 2/(x_i-x_j)^2 + V''(x_i)
        J = np.zeros((n, n))
        Vpp = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if j != i:
                    J[i, j] = -2.0 / (x[i] - x[j])**2
            # V''(x) = 2*a1/(x-c1)^3 + 2*a2/(x-c2)^3 + 2*a3/(x-c3)^3
            Vpp_i = 2.0*a1/(x[i]-c1)**3 + 2.0*a2/(x[i]-c2)**3 + 2.0*a3/(x[i]-c3)**3
            J[i, i] = -2.0 * sum(1.0/(x[i]-x[j])**2 for j in range(n) if j != i) + Vpp_i
        
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            try:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]
            except:
                break
        
        # Damped Newton step
        alpha_step = 1.0
        for _ in range(20):
            x_new = x + alpha_step * dx
            if np.all(x_new > 0) and np.all(np.isfinite(x_new)):
                r_new = stationarity_residual(x_new)
                if np.max(np.abs(r_new)) < np.max(np.abs(r)):
                    x = x_new
                    break
            alpha_step *= 0.5
        else:
            break
    
    # Sort and ensure positivity
    x = np.sort(x)
    x = np.maximum(x, 1e-10)
    return x

def _rational_potential_objective(params):
    """Full objective: solve for roots given potential params, then compute C4."""
    try:
        # Use base roots as initial guess
        xg, _ = roots_laguerre(12)
        base_roots = np.sort(xg)
        
        roots = _rational_potential_roots(params, base_roots)
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return np.inf
        return _compute_c4(roots)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: Coarse-to-fine 1D search over alpha for generalized Laguerre
    best_alpha = 0.0
    best_c4_alpha = np.inf

    # Coarse grid
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 60)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _c4_from_alpha(np.exp(la))
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Fine grid around best
    if time_ok():
        local_lo = math.log(max(best_alpha * 0.5, 1e-3))
        local_hi = math.log(min(best_alpha * 2.0, 1e4))
        if local_hi > local_lo:
            log_alphas_fine = np.linspace(local_lo, local_hi, 250)
            for la in log_alphas_fine:
                if not time_ok():
                    break
                c4_val = _c4_from_alpha(np.exp(la))
                if c4_val < best_c4_alpha:
                    best_c4_alpha = c4_val
                    best_alpha = np.exp(la)

    # Scalar minimization
    if time_ok():
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(1e-3), math.log(1e4)),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 300}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get base roots from best alpha
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Also try standard Laguerre
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: Calogero-Sutherland equilibrium initializations
    if time_ok():
        beta_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        for beta in beta_values:
            if not time_ok():
                break
            try:
                cs_roots = _calogero_sutherland_roots(beta)
                if np.all(cs_roots > 0) and np.all(np.isfinite(cs_roots)):
                    c4_cs = _compute_c4(cs_roots)
                    if c4_cs < best_c4:
                        best_c4 = c4_cs
                        best_roots = cs_roots.copy()
            except Exception:
                pass

    # Phase 3: Full 12-DOF L-BFGS-B refinement from best
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start with diverse initializations
    if time_ok():
        rng = np.random.default_rng(42)
        
        # Perturbed best
        for trial in range(6):
            if not time_ok():
                break
            pert = best_roots * (1.0 + 0.15 * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            try:
                res = minimize(
                    _objective_grad,
                    pert,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-8, None)] * 12,
                    options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 40}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass
        
        # Different alpha values
        for alpha_offset in [0.5, 2.0, 0.25, 4.0, 0.1, 10.0]:
            if not time_ok():
                break
            for sign in [-1, 1]:
                a_try = best_alpha * alpha_offset if sign > 0 else best_alpha / alpha_offset
                if a_try < 1e-3 or a_try > 1e4:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    res2 = minimize(
                        _objective_grad,
                        xg2,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 40}
                    )
                    if np.all(res2.x > 0) and np.isfinite(res2.fun):
                        candidate = np.sort(res2.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 5: Log-space polish
    if time_ok():
        try:
            log_roots_init = np.log(best_roots)
            res_log = minimize(
                _exp_objective_grad,
                log_roots_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 50}
            )
            if np.isfinite(res_log.fun):
                candidate = np.sort(np.exp(res_log.x))
                if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 6: Final high-precision polish (multiple alternating passes)
    for polish_iter in range(12):
        if not time_ok():
            break
        # Direct space polish
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-12, None)] * 12,
                options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass
        # Log-space polish
        if not time_ok():
            break
        try:
            log_roots_init = np.log(best_roots)
            res_log = minimize(
                _exp_objective_grad,
                log_roots_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 50}
            )
            if np.isfinite(res_log.fun):
                candidate = np.sort(np.exp(res_log.x))
                if np.all(candidate > 0) and np.all(np.isfinite(candidate)):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
