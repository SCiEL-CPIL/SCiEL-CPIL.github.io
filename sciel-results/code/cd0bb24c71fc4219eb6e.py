"""Search for optimal Hermite coefficients minimizing r^2/(2*pi)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

# Deadline guard: evaluator kills at 210s; use 190s as hard limit with margin
_DEADLINE = time.time() + 185.0

def _time_left():
    return _DEADLINE - time.time()


def _hermite_monomial_coeffs(n):
    """Return monomial coefficients [c0, c1, ..., cn] for physicists' Hermite H_n(x).
    H_n(x) = sum_{k} c_k x^k where we return c_k for k=0..n."""
    # Use the explicit formula:
    # H_n(x) = n! * sum_{m=0}^{floor(n/2)} (-1)^m (2x)^(n-2m) / (m! (n-2m)!)
    coeffs = np.zeros(n + 1)
    for m in range(n // 2 + 1):
        power = n - 2 * m
        c = (-1)**m * factorial(n) / (factorial(m) * factorial(power))
        coeffs[power] = c
    return coeffs


# Precompute monomial coefficients for H_0, H_4, H_8, H_12
_H0_mono = _hermite_monomial_coeffs(0)  # [1]
_H4_mono = _hermite_monomial_coeffs(4)  # degree 4
_H8_mono = _hermite_monomial_coeffs(8)  # degree 8
_H12_mono = _hermite_monomial_coeffs(12)  # degree 12


def _build_q_coeffs(c0, c1, c2):
    """
    Build the monomial coefficients of q(x) = g(x)/x^2 where
    g = c0*H0 + c1*H4 + c2*H8 + c3*H12, with c3 chosen so g(0)=0.
    
    H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-1032160
    c3 = -(c0*1 + c1*(-12) + c2*1680) / (-1032160)
       = (c0 - 12*c1 + 1680*c2) / 1032160
    
    g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    q(x) = g(x)/x^2, which is a polynomial of degree 10 (since g has no constant or linear term).
    
    We compute monomial coefficients of g, then divide by x^2 (shift down by 2).
    """
    H0_0 = 1
    H4_0 = -12
    H8_0 = 1680
    H12_0 = -1032160
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    c3 = (c0 - 12*c1 + 1680*c2) / 1032160  # simplified
    
    # Build g monomial coefficients (degree 12)
    g_coeffs = np.zeros(13)
    for i, c in enumerate(_H0_mono):
        g_coeffs[i] += c0 * c
    for i, c in enumerate(_H4_mono):
        g_coeffs[i] += c1 * c
    for i, c in enumerate(_H8_mono):
        g_coeffs[i] += c2 * c
    for i, c in enumerate(_H12_mono):
        g_coeffs[i] += c3 * c
    
    # g_coeffs[0] should be 0 (by construction), g_coeffs[1] should be 0 (all H_n for n>=2 have no linear term, H0 is constant)
    # Actually H0 has no linear term, H4, H8, H12 are even so no odd terms. So g_coeffs[1]=0 automatically.
    
    # q(x) = g(x)/x^2: shift coefficients down by 2
    # q has degree 10, coefficients q_coeffs[j] = g_coeffs[j+2] for j=0..10
    q_coeffs = g_coeffs[2:13]  # 11 coefficients for degree 10 polynomial
    
    return q_coeffs


def _find_largest_sign_change_root(q_coeffs):
    """
    Find the largest real root of q at which q changes sign.
    q_coeffs: array of monomial coefficients [a0, a1, ..., a10] for q(x) = a0 + a1*x + ... + a10*x^10
    """
    # Use np.roots to find all roots
    roots = np.roots(q_coeffs)
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(float(r.real))
    
    if not real_roots:
        return None
    
    # For each real root, check if q changes sign there.
    # A root changes sign if it has odd multiplicity.
    # Numerically: evaluate q at root +/- epsilon and check sign change.
    # Better: use the fact that for a simple root, sign changes. For multiplicity m, sign changes iff m is odd.
    
    # Let's evaluate q at points near each root to detect sign change
    eps = 1e-8
    best_root = None
    
    for r in real_roots:
        # Evaluate q at r - eps and r + eps
        # Use Horner's method
        def q_eval(x):
            result = 0.0
            for c in reversed(q_coeffs):
                result = result * x + c
            return result
        
        q_minus = q_eval(r - eps)
        q_plus = q_eval(r + eps)
        
        if q_minus * q_plus < 0:  # sign change
            if best_root is None or r > best_root:
                best_root = r
    
    return best_root


def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients. Returns inf if invalid."""
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    r = _find_largest_sign_change_root(q_coeffs)
    if r is None:
        return np.inf
    return r**2 / (2 * np.pi)


def _objective(params):
    """Objective function for optimization: minimize value."""
    c0, c1, c2 = params
    val = _compute_value(c0, c1, c2)
    return val


def construct():
    """Find three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_params = seed.copy()
    best_value = _compute_value(seed[0], seed[1], seed[2])
    
    if np.isinf(best_value):
        # Seed is invalid? Shouldn't happen, but handle gracefully
        return seed
    
    # Multi-start Nelder-Mead search
    n_starts = 15
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points around the seed and in broader regions
    start_points = [seed.copy()]
    
    # Small perturbations around seed
    for _ in range(5):
        perturbation = rng.normal(0, 0.05, size=3)
        start_points.append(seed + perturbation)
    
    # Medium perturbations
    for _ in range(5):
        perturbation = rng.normal(0, 0.2, size=3)
        start_points.append(seed + perturbation)
    
    # Broader exploration
    for _ in range(4):
        p = np.array([rng.uniform(-2, 2), rng.uniform(-0.5, 0.5), rng.uniform(-0.01, 0.01)])
        start_points.append(p)
    
    for i, x0 in enumerate(start_points):
        if _time_left() < 10.0:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Polish: high-precision refinement around the best point found
    if _time_left() > 5.0 and not np.isinf(best_value):
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final sanity check: if something went wrong, return seed
    if np.isinf(best_value) or np.isnan(best_value):
        return seed
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
