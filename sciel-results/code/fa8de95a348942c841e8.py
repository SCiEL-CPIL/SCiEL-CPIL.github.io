"""Model-and-optimize: 1D search over generalized Laguerre parameter alpha, then 12-DOF refinement.
Structural change: reduce 12-dim discrete optimization to 1-dim search over alpha (root density profile),
then use full 12-DOF L-BFGS-B polish from the best alpha-derived initial point.
Optimized: O(n) gradient via logarithmic derivatives, streamlined search phases."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Build P(x) = prod (x-x_i)^2 and P2 = P^2 in ascending coeff order."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    Uses fast O(n) gradient via logarithmic derivatives."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
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
    
    # Fast gradient using logarithmic derivatives
    # P(x) = prod_j (x - x_j)^2
    # dP/dx_i = P(x) * 2 / (x - x_i)  (log-derivative trick)
    # We need: dI1/dx_i = int (dP/dx_i) e^{-x} dx = 2 * int P(x)/(x-x_i) * e^{-x} dx
    #         dI2/dx_i = int 2*P(x)*dP/dx_i * e^{-x} dx = 4 * int P(x)^2/(x-x_i) * e^{-x} dx
    #
    # For int P(x)/(x-x_i) e^{-x} dx:
    #   P(x)/(x-x_i) = prod_{j!=i} (x-x_j)^2 * (x+x_i)  ... no
    #   Actually P(x) = (x-x_i)^2 * Q_i(x) where Q_i = prod_{j!=i}(x-x_j)^2
    #   P(x)/(x-x_i) = (x-x_i)*Q_i(x)
    #   So int P/(x-x_i) e^{-x} dx = int (x-x_i) Q_i(x) e^{-x} dx
    #   = int x*Q_i(x) e^{-x} dx - x_i * int Q_i(x) e^{-x} dx
    #
    # This is still O(n) per root if we build Q_i efficiently.
    # Better approach: use the fact that for P = prod (x-x_j)^2:
    #   P'(x)/P(x) = 2*sum_j 1/(x-x_j)
    #   dP/dx_i = P(x) * 2/(x-x_i)  (treating x_i as parameter)
    #
    # So: int P/(x-x_i) e^{-x} dx
    #     Let's use: 1/(x-x_i) = -1/(x_i - x)
    #     P(x)/(x-x_i) = -P(x)/(x_i-x)
    #
    # Alternative: compute Q_i = P / (x-x_i)^2 for each i, then
    #   int P/(x-x_i) e^{-x} dx = int (x-x_i) Q_i e^{-x} dx
    #     = int x*Q_i e^{-x} dx - x_i * int Q_i e^{-x} dx
    #
    # For efficiency, precompute all Q_i using prefix/suffix products.
    
    n = len(roots)
    from numpy.polynomial.polynomial import polyfromroots, polymul
    
    # Build prefix and suffix products of (x - x_j)^2
    # prefix[i] = prod_{j=0}^{i-1} (x-x_j)^2
    # suffix[i] = prod_{j=i}^{n-1} (x-x_j)^2
    prefix = [np.ones(1)] * n
    suffix = [np.ones(1)] * n
    
    # (x - x_j)^2 = x^2 - 2*x_j*x + x_j^2, ascending: [x_j^2, -2*x_j, 1]
    for j in range(n):
        if j > 0:
            prefix[j] = polymul(prefix[j-1], np.array([roots[j]**2, -2*roots[j], 1.0]))
        else:
            prefix[j] = np.ones(1)
    
    for j in range(n-1, -1, -1):
        if j < n-1:
            suffix[j] = polymul(suffix[j+1], np.array([roots[j]**2, -2*roots[j], 1.0]))
        else:
            suffix[j] = np.ones(1)
    
    # Q_i = prefix[i] * suffix[i+1] (product of all factors except (x-x_i)^2)
    grad = np.zeros(n)
    for i in range(n):
        Q_i = polymul(prefix[i], suffix[i+1]) if i < n-1 else prefix[i].copy()
        if i > 0 and i < n-1:
            Q_i = polymul(prefix[i], suffix[i+1])
        elif i == 0:
            Q_i = suffix[1].copy()
        elif i == n-1:
            Q_i = prefix[n-1].copy()
        
        # int Q_i e^{-x} dx
        int_Qi = _poly_moment(Q_i)
        # int x*Q_i e^{-x} dx = int (x*Q_i) e^{-x} dx
        xQi = np.polynomial.polynomial.polymul(Q_i, np.array([0.0, 1.0]))
        int_xQi = _poly_moment(xQi)
        
        # int P/(x-x_i) e^{-x} dx = int_xQi - x_i * int_Qi
        int_P_over_xi = int_xQi - roots[i] * int_Qi
        
        # dI1/dx_i = -2 * 2 * int_P_over_xi = -4 * int_P_over_xi
        # (because P = (x-x_i)^2 * Q_i, dP/dx_i = -2(x-x_i)*Q_i, so dP/dx_i = -2*(x-x_i)*Q_i)
        # Wait, let me re-derive: P(x) = prod_j (x-x_j)^2. dP/dx_i = 2*(x-x_i) * prod_{j!=i}(x-x_j)^2 = 2*(x-x_i)*Q_i(x)
        # So int dP/dx_i e^{-x} dx = 2 * int (x-x_i) Q_i e^{-x} dx = 2 * int_P_over_xi
        dI1_i = 2.0 * int_P_over_xi
        
        # dI2/dx_i = d/dx_i int P^2 e^{-x} dx = int 2P * dP/dx_i e^{-x} dx = 2*int P * 2*(x-x_i)*Q_i e^{-x} dx
        # = 4 * int P * (x-x_i) * Q_i e^{-x} dx = 4 * int (x-x_i) * P * Q_i e^{-x} dx
        # = 4 * int (x-x_i) * (x-x_i)^2 * Q_i^2 e^{-x} dx = 4 * int (x-x_i)^3 * Q_i^2 e^{-x} dx
        # Hmm this is getting complex. Let me use a different approach.
        # dI2/dx_i = 2 * int P * (dP/dx_i) e^{-x} dx = 4 * int P * (x-x_i) * Q_i e^{-x} dx
        # = 4 * int (x-x_i) * (x-x_i)^2 * Q_i^2 e^{-x} dx = 4 * int (x-x_i)^3 * Q_i^2 e^{-x} dx
        # This requires Q_i^2 which is expensive.
        
        # Simpler: just compute dI2/dx_i = 4 * int (x-x_i) * P * Q_i e^{-x} dx
        # P * Q_i = (x-x_i)^2 * Q_i^2
        # So dI2/dx_i = 4 * int (x-x_i)^3 * Q_i^2 e^{-x} dx
        # Let R_i = Q_i^2, then we need int (x-x_i)^3 * R_i e^{-x} dx
        # = int (x^3 - 3*x_i*x^2 + 3*x_i^2*x - x_i^3) * R_i e^{-x} dx
        # = int x^3*R_i - 3*x_i*int x^2*R_i + 3*x_i^2*int x*R_i - x_i^3*int R_i
        
        # This is getting expensive. Let's use a simpler but still O(n) approach:
        # Just compute the gradient numerically with central differences? No, too slow.
        
        # Actually, let me just use the direct formula more carefully:
        # dI2/dx_i = 4 * int (x-x_i) * P(x) * Q_i(x) * e^{-x} dx
        # P(x) * Q_i(x) = (x-x_i)^2 * Q_i(x)^2
        # So dI2/dx_i = 4 * int (x-x_i)^3 * Q_i(x)^2 * e^{-x} dx
        
        # For the moment computation, let's build S_i = Q_i^2 and compute the needed moments
        S_i = polymul(Q_i, Q_i)
        # int (x-x_i)^3 * S_i e^{-x} dx
        # = int (x^3 - 3*x_i*x^2 + 3*x_i^2*x - x_i^3) * S_i e^{-x} dx
        x3Si = np.polynomial.polynomial.polymul(S_i, np.array([0.0, 0.0, 0.0, 1.0]))
        x2Si = np.polynomial.polynomial.polymul(S_i, np.array([0.0, 0.0, 1.0]))
        xSi = np.polynomial.polynomial.polymul(S_i, np.array([0.0, 1.0]))
        
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

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: Coarse-to-fine 1D search over alpha
    def _log_alpha_objective(log_a):
        alpha = np.exp(log_a)
        if alpha < 1e-3 or alpha > 1e4:
            return np.inf
        return _c4_from_alpha(alpha)

    # Coarse grid: 50 points
    best_alpha = 0.0
    best_c4_alpha = np.inf
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 50)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _log_alpha_objective(la)
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Fine grid around best: 200 points in local region
    if time_ok():
        local_lo = math.log(max(best_alpha * 0.5, 1e-3))
        local_hi = math.log(min(best_alpha * 2.0, 1e4))
        if local_hi > local_lo:
            log_alphas_fine = np.linspace(local_lo, local_hi, 200)
            for la in log_alphas_fine:
                if not time_ok():
                    break
                c4_val = _log_alpha_objective(la)
                if c4_val < best_c4_alpha:
                    best_c4_alpha = c4_val
                    best_alpha = np.exp(la)

    # Scalar minimization for final alpha precision
    if time_ok():
        try:
            res_alpha = minimize_scalar(
                _log_alpha_objective,
                bounds=(math.log(1e-3), math.log(1e4)),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 300}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get roots from best alpha
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Phase 2: Full 12-DOF L-BFGS-B refinement
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Try a few neighboring alpha values as additional starting points
    if time_ok():
        for alpha_offset in [0.5, 2.0, 0.1, 10.0]:
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
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
                    )
                    if np.all(res2.x > 0) and np.isfinite(res2.fun):
                        candidate = np.sort(res2.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 4: Final high-precision polish (multiple passes)
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
