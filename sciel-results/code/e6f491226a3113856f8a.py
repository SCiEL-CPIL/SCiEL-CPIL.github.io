"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_DEADLINE = 210.0
_START = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START) - 5.0  # 5s safety margin

def _compute_value(c0, c1, c2):
    """Compute the exact value using sympy rational arithmetic."""
    import sympy as sp
    from sympy.polys.orthopolys import hermite_poly

    x = sp.Symbol('x')
    H0 = hermite_poly(0, x)
    H4 = hermite_poly(4, x)
    H8 = hermite_poly(8, x)
    H12 = hermite_poly(12, x)

    # Convert floats to rationals
    c0_r = Fraction(c0).limit_denominator(10**12)
    c1_r = Fraction(c1).limit_denominator(10**12)
    c2_r = Fraction(c2).limit_denominator(10**12)

    # Compute c3 so that g(0) = 0
    H0_0 = H0.subs(x, 0)
    H4_0 = H4.subs(x, 0)
    H8_0 = H8.subs(x, 0)
    H12_0 = H12.subs(x, 0)

    c3_r = -(c0_r * H0_0 + c1_r * H4_0 + c2_r * H8_0) / H12_0

    g = c0_r * H0 + c1_r * H4 + c2_r * H8 + c3_r * H12

    # Check if g tends to negative limit at infinity (leading coefficient negative)
    # H12 leading term: 12! / (12/2)! * x^12 ... actually hermite_poly uses physicists' convention
    # Leading coefficient of H_n is 2^n
    # g's leading coeff is c3_r * 2^12 = c3_r * 4096
    if c3_r < 0:
        g = -g
        c3_r = -c3_r

    # q = g / x^2
    q = sp.simplify(g / x**2)

    # Find all real roots of q
    roots = sp.roots(q, x)
    real_roots = []
    for root, mult in roots.items():
        if root.is_real:
            real_roots.append((root, mult))

    # Find roots where q changes sign (odd multiplicity)
    sign_change_roots = []
    for root, mult in real_roots:
        if mult % 2 == 1:
            sign_change_roots.append(float(root))

    if not sign_change_roots:
        return float('inf')

    r = max(sign_change_roots)
    return r**2 / (2 * np.pi)

def _objective(c):
    """Objective function with non-negativity handling."""
    c0, c1, c2 = c
    try:
        return _compute_value(c0, c1, c2)
    except Exception:
        return 1e10

def construct():
    import numpy as np
    from scipy.optimize import minimize

    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best_c = seed.copy()
    best_val = _objective(seed)

    # Nelder-Mead local search
    x0 = seed
    result = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
    if result.fun < best_val:
        best_val = result.fun
        best_c = result.x

    # Try a few different starting points
    rng = np.random.default_rng(42)
    for i in range(10):
        if _time_remaining() < 10:
            break
        # Perturb the seed
        perturbation = rng.normal(0, 0.1, size=3)
        start = seed + perturbation
        result = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_c = result.x

    # Final polish with tighter tolerance
    if _time_remaining() > 5:
        result = minimize(_objective, best_c, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_c = result.x

    return np.array(best_c)
# EVOLVE-BLOCK-END
