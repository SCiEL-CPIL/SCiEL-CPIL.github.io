"""Optimize coefficients to minimize the largest positive sign-changing root of q(y)=g(x)/x^2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Hermite polynomials H_0, H_4, H_8, H_12 as polynomials in y = x^2
# H_n(x) expressed in terms of y = x^2 (even part only, since g must be even for q to be polynomial)
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12  => 16y^2 - 48y + 12
# H_8(x) = 256y^4 - 1344y^3 + 2688y^2 - 1344y + 112
# H_12(x) = 4096y^6 - 46080y^5 + 205824y^4 - 387072y^3 + 322560y^2 - 137280y + 137280

_H0_y = np.array([0, 0, 0, 0, 0, 0, 1])          # y^6..y^0
_H4_y = np.array([0, 0, 0, 0, 16, -48, 12])
_H8_y = np.array([0, 0, 256, -1344, 2688, -1344, 112])
_H12_y = np.array([4096, -46080, 205824, -387072, 322560, -137280, 137280])

_CONST = 137280.0  # H12(0)


def _build_q_coeffs(c0, c1, c2):
    """Return coefficients of q(y) = g(x)/x^2 as a polynomial in y (degree 5).
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12, with c3 chosen so g(0)=0.
    q(y) = g(sqrt(y))/y. Since g is even, g(x)/x^2 is a polynomial in y of degree 5."""
    # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0)
    # H0(0)=1, H4(0)=12, H8(0)=112, H12(0)=137280
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 112.0) / _CONST
    
    # g(x) as polynomial in x (even powers only)
    # g(x) = c0*(1) + c1*(16x^4-48x^2+12) + c2*(256x^8-1344x^6+2688x^4-1344x^2+112) + c3*(4096x^12-46080x^10+205824x^8-387072x^6+322560x^4-137280x^2+137280)
    # q(y) = g(sqrt(y))/y = [coeffs of x^2, x^4, ..., x^12] / y
    # So q(y) has degree 5 with coefficients from x^2 to x^12 terms
    
    # Collect coefficients of x^{2k} for k=0..6 (k=0 is constant, which should be 0 since g(0)=0)
    # g(x) = sum_{k=0}^{6} a_k x^{2k}, with a_0 = 0
    # q(y) = sum_{k=0}^{5} a_{k+1} y^k
    
    a0 = c0 * 1 + c1 * 12 + c2 * 112 + c3 * 137280  # should be ~0
    a1 = c1 * (-48) + c2 * (-1344) + c3 * (-137280)
    a2 = c1 * 16 + c2 * 2688 + c3 * 322560
    a3 = c2 * (-1344) + c3 * (-387072)
    a4 = c2 * 256 + c3 * 205824
    a5 = c3 * (-46080)
    a6 = c3 * 4096
    
    # q(y) = a1 + a2*y + a3*y^2 + a4*y^3 + a5*y^4 + a6*y^5
    # But we want it as numpy poly format: [a6, a5, a4, a3, a2, a1] for y^5..y^0
    q_coeffs = np.array([a6, a5, a4, a3, a2, a1], dtype=float)
    return q_coeffs


def _largest_positive_sign_change_root(q_coeffs, penalty_factor=1000.0):
    """Find the largest positive real root of q(y) at which q changes sign.
    Returns a large penalty if no such root exists or if constraints are violated."""
    # Trim leading zeros
    nz = np.where(np.abs(q_coeffs) > 1e-15 * max(1.0, np.max(np.abs(q_coeffs))))[0]
    if len(nz) == 0:
        return 1e18
    q_coeffs = q_coeffs[nz[0]:]
    
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return 1e18
    
    best = -1.0
    penalty = 0.0
    
    for rt in roots:
        if abs(rt.imag) > 1e-8 * max(1.0, abs(rt.real)):
            continue
        r = rt.real
        if r <= 1e-12:
            continue
        # Check sign change
        d = 1e-5 * max(1.0, abs(r))
        try:
            left = np.polyval(q_coeffs, r - d)
            right = np.polyval(q_coeffs, r + d)
        except Exception:
            continue
        if left * right < 0.0:
            if r > best:
                best = r
        else:
            # Penalize roots that don't have a strict sign change
            penalty += penalty_factor * (abs(left * right) + 1e-10)
    
    # Check leading coefficient sign (should be positive after potential flip)
    if len(q_coeffs) > 0 and q_coeffs[0] < 0:
        penalty += penalty_factor * abs(q_coeffs[0])
    
    if best > 0.0:
        return best + penalty
    else:
        return 1e18


def _objective(params):
    """Objective: largest positive sign-changing root of q(y). Lower is better."""
    c0, c1, c2 = params
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    return _largest_positive_sign_change_root(q_coeffs)


def _objective_ratio(ratios):
    """Objective in ratio space: c1/c0, c2/c0 with c0=1."""
    c1, c2 = ratios
    return _objective(np.array([1.0, c1, c2]))


def _seed_params():
    return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])


def construct():
    t0 = time.time()
    deadline = 195.0  # safety margin under 210s
    
    seed = _seed_params()
    best_params = seed.copy()
    best_val = _objective(best_params)
    
    # Seed in ratio space
    seed_ratio = np.array([seed[1]/seed[0], seed[2]/seed[0]])
    
    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 60:
        try:
            bounds = [(-5.0, 5.0), (-0.5, 0.5)]
            de_res = differential_evolution(
                _objective_ratio, bounds,
                seed=42, maxiter=200, popsize=15,
                tol=1e-12, mut=(0.5, 1.0), recomb=0.7,
                polish=False, init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = np.array([1.0, de_res.x[0], de_res.x[1]])
        except Exception:
            pass
    
    # Phase 2: Multi-start Nelder-Mead from best DE result and seed
    start_points = [best_params[1:].copy(), seed_ratio.copy()]
    rng = np.random.default_rng(42)
    for _ in range(10):
        start_points.append(best_params[1:] + rng.normal(0, 0.3, size=2))
    
    start_scores = []
    for s in start_points:
        if time.time() > deadline - 30:
            break
        val = _objective_ratio(s)
        start_scores.append((val, s))
    
    start_scores.sort(key=lambda x: x[0])
    
    # Run Nelder-Mead from the top 3 starting points
    for i, (val, s) in enumerate(start_scores[:3]):
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(_objective_ratio, s, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14,
                                  'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = np.array([1.0, res.x[0], res.x[1]])
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective_ratio, best_params[1:], method='Nelder-Mead',
                         options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-16,
                                  'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = np.array([1.0, res.x[0], res.x[1]])
        except Exception:
            pass
    
    # Safety: if seed is better, use it
    seed_val = _objective(seed)
    if seed_val < best_val:
        best_params = seed
    
    return [float(best_params[0]), float(best_params[1]), float(best_params[2])]
# EVOLVE-BLOCK-END
