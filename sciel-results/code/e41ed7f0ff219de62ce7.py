"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Uses rational potential V(x) = (a1*x^2 + a2*x + a3)/(x+b) critical points as a 4D manifold
of root configurations, combined with direct L-BFGS-B optimization on the full 12D space."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly * e^{-x} dx over [0,inf). Coeffs in ascending order."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:min(n, 60)], _FACTS[:min(n, 60)]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
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
    
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        R_i = np.ones(1)
        for j in range(len(roots)):
            if j == i:
                continue
            R_i = np.polynomial.polynomial.polymul(R_i, np.polynomial.polynomial.polyfromroots([roots[j], roots[j]]))
        xR_i = np.polynomial.polynomial.polymul(R_i, np.array([0.0, 1.0]))
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
        RP = np.polynomial.polynomial.polymul(R_i, P)
        xRP = np.polynomial.polynomial.polymul(RP, np.array([0.0, 1.0]))
        term2 = _poly_moment(xRP) - roots[i] * _poly_moment(RP)
        dI2_i = -4.0 * term2
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _rational_potential_roots(params):
    """Given (a1, a2, a3, b), find critical points of V(x) = (a1*x^2 + a2*x + a3)/(x+b).
    V'(x) = 0 gives a quadratic in x. We need 12 roots, so this only gives 2.
    Instead, use V(x) = (a1*x^4 + a2*x^3 + a3*x^2 + a4*x + a5)/(x+b) for 5 critical points,
    or better: use a degree-6 numerator for 12 critical points... that's too many params.
    
    Alternative: Use V(x) = sum_{k=1}^{4} c_k / (x - p_k) with 4 poles and 4 coefficients = 8 params,
    then V'(x) = 0 gives a degree-8 polynomial -> 8 roots. Not 12.
    
    Let's use V(x) = sum_{k=1}^{6} c_k / (x - p_k) with 6 poles. That's 12 params - too many.
    
    Better approach: V(x) = (x^6 + s1*x^5 + s2*x^4 + s3*x^3 + s4*x^2 + s5*x + s6) / (x + b)
    This has 7 free params (s1..s6, b). V'(x) = 0 gives a degree-9 polynomial -> 9 roots. Still not 12.
    
    Let me try: V(x) = (x^8 + s1*x^7 + ... + s8) / (x + b) -> 9 params, V'(x)=0 gives degree 9 -> 9 roots.
    
    For 12 roots from V'=0, we need numerator degree 13, denominator degree 1: 14 params. Too many.
    
    Let's use a different structure: V(x) = (a*x^12 + b*x^11 + ... ) / (x+c) - too many params.
    
    Actually, let's just use a 4-param family: V(x) = (x^2 + a*x + b)/(x + c) doesn't work for 12 roots.
    
    New idea: Use the fact that the optimal roots might be well-approximated by a scaled/shifted
    version of known orthogonal polynomial roots. Let's parameterize as:
    x_i = scale * y_i + shift, where y_i are roots of some base polynomial.
    But that's only 2 params.
    
    Let's just do multi-start L-BFGS-B with diverse initializations."""
    return None

def _c4_from_alpha(alpha):
    """Get roots from generalized Laguerre and compute C4."""
    try:
        nodes, _ = roots_genlaguerre(12, alpha)
        nodes = np.sort(nodes)
        if np.any(nodes <= 0):
            return None, None
        c4 = _compute_c4(nodes)
        if np.isfinite(c4):
            return c4, nodes
    except Exception:
        pass
    return None, None

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    best_roots = None
    best_c4 = np.inf

    # Phase 1: Broad alpha scan
    alpha_values = list(np.linspace(-5.0, 5.0, 51))
    alpha_values.extend([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 0.01, -0.01,
                         3.0, -3.0, 4.0, -4.0, 1.5, -1.5, 0.25, -0.25])
    alpha_values = sorted(set(alpha_values))

    for alpha in alpha_values:
        if not time_ok():
            break
        c4, nodes = _c4_from_alpha(alpha)
        if c4 is not None and c4 < best_c4:
            best_c4 = c4
            best_roots = nodes.copy()

    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Phase 2: Multi-start L-BFGS-B from diverse initializations
    rng = np.random.RandomState(42)
    starts = [best_roots.copy()]
    
    # Add perturbed versions of best
    for scale in [0.8, 0.9, 1.1, 1.2]:
        starts.append(best_roots * scale)
    
    # Add random positive starts
    for _ in range(5):
        r = np.sort(rng.uniform(0.1, 50.0, 12))
        starts.append(r)
    
    # Add starts from different alpha values
    for alpha in [0.0, 1.0, -1.0, 2.0, 0.5]:
        c4_a, nodes_a = _c4_from_alpha(alpha)
        if nodes_a is not None:
            starts.append(nodes_a.copy())
    
    # Add structured starts: geometric spacing
    for base in [1.0, 2.0, 0.5]:
        r = np.sort(base * np.exp(np.linspace(0, 3, 12)))
        starts.append(r)
    
    # Add starts with clustering
    for center in [5.0, 10.0, 1.0]:
        r = np.sort(center + rng.normal(0, center*0.3, 12))
        r = np.maximum(r, 0.01)
        starts.append(r)

    for s_idx, s in enumerate(starts):
        if not time_ok():
            break
        if np.any(s <= 0) or not np.all(np.isfinite(s)):
            continue
        try:
            res = minimize(
                _objective_grad,
                s.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 40}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Fine alpha scan around best region + refinement
    if time_ok():
        best_alpha = 0.0
        best_alpha_c4 = np.inf
        for alpha in alpha_values:
            c4_a, nodes_a = _c4_from_alpha(alpha)
            if c4_a is not None and c4_a < best_alpha_c4:
                best_alpha_c4 = c4_a
                best_alpha = alpha
        
        fine_alphas = np.linspace(best_alpha - 0.3, best_alpha + 0.3, 31)
        for alpha in fine_alphas:
            if not time_ok():
                break
            c4_a, nodes_a = _c4_from_alpha(alpha)
            if c4_a is not None:
                if c4_a < best_c4:
                    best_c4 = c4_a
                    best_roots = nodes_a.copy()
                try:
                    res = minimize(
                        _objective_grad,
                        nodes_a.copy(),
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 40}
                    )
                    if np.all(res.x > 0) and np.isfinite(res.fun):
                        candidate = np.sort(res.x)
                        c4 = _compute_c4(candidate)
                        if np.isfinite(c4) and c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 4: High-precision polish
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 5: Additional local search from best with small perturbations
    if time_ok():
        for _ in range(3):
            if not time_ok():
                break
            perturbed = best_roots + rng.normal(0, 0.01 * np.mean(best_roots), 12)
            perturbed = np.maximum(perturbed, 1e-6)
            try:
                res = minimize(
                    _objective_grad,
                    perturbed.copy(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-10, None)] * 12,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
