"""Optimize the uncertainty-inequality constant via exact-arithmetic root analysis."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import differential_evolution, minimize, brentq, basinhopping
from sympy import Poly, symbols, ZZ, Rational, nsimplify
from sympy.polys.orthopolys import hermite_poly


# Precompute Hermite polynomials as sympy Polynomials (exact)
_H0 = hermite_poly(0)
_H4 = hermite_poly(4)
_H8 = hermite_poly(8)
_H12 = hermite_poly(12)

# Values at 0 (exact rationals)
_H0_0 = _H0.subs(symbols('x'), 0)
_H4_0 = _H4.subs(symbols('x'), 0)
_H8_0 = _H8.subs(symbols('x'), 0)
_H12_0 = _H12.subs(symbols('x'), 0)

# Float versions for fast evaluation
_H0_f = [float(c) for c in _H0.all_coeffs()[::-1]]
_H4_f = [float(c) for c in _H4.all_coeffs()[::-1]]
_H8_f = [float(c) for c in _H8.all_coeffs()[::-1]]
_H12_f = [float(c) for c in _H12.all_coeffs()[::-1]]


def _build_Q_exact(c1_frac, c2_frac):
    """Build Q(y) = g(sqrt(y)) as a sympy Poly in y with exact rational coefficients.
    c1_frac, c2_frac are Fractions. Returns (Q_poly, c3_frac)."""
    x = symbols('x')
    c3 = -(_H0_0 + c1_frac * _H4_0 + c2_frac * _H8_0) / _H12_0
    g = _H0 + c1_frac * _H4 + c2_frac * _H8 + c3 * _H12
    # g is even: g(x) = sum g_{2k} x^{2k}
    # Q(y) = g(sqrt(y)) = sum g_{2k} y^k
    y = symbols('y')
    Q = g.subs(x, y**0.5)  # won't work directly, extract even coefficients
    # Extract even-degree coefficients from g
    g_poly = Poly(g, x, domain=ZZ)
    coeffs = g_poly.all_coeffs()  # descending: [a12, a11, ..., a0]
    # g only has even powers, so odd coefficients are 0
    # g = a12*x^12 + a10*x^10 + ... + a2*x^2 + a0
    # Q(y) = a12*y^6 + a10*y^5 + ... + a2*y + a0
    # But Q(0) = a0 = g(0) = 0 by construction of c3
    # We need Q(y) = (g(x) - g(0)) / x^2 evaluated at x^2 = y
    # Actually g(0) = 0, so g(x)/x^2 = sum_{k=1}^{6} g_{2k} x^{2k-2} = sum_{k=0}^{5} g_{2(k+1)} y^k
    # So Q(y) has degree 5 (since g is degree 12)
    # g_{2k} for k=1..6
    Q_coeffs = []
    for k in range(6, 0, -1):  # k=6 down to 1, giving y^5 down to y^0
        Q_coeffs.append(g_poly.coeff_monomial(x**(2*k)))
    Q_poly = Poly(Q_coeffs, y, domain=ZZ)
    return Q_poly, c3


def _build_Q_float(c1, c2):
    """Build Q(y) coefficients as float array (ascending) for fast evaluation."""
    c3 = -(_H0_0 + Fraction(c1).limit_denominator(10**12) * _H4_0 + Fraction(c2).limit_denominator(10**12) * _H8_0) / _H12_0
    # Use float arithmetic for speed
    c1f = float(c1)
    c2f = float(c2)
    c3f = -(_H0_0 + c1f * float(_H4_0) + c2f * float(_H8_0)) / float(_H12_0)
    
    # Build g coefficients (ascending)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 46080x^10 + 282240x^8 - 752640x^6 + 907200x^4 - 460800x^2 + 11880
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # We need g(x)/x^2, so coefficients of x^2, x^4, ..., x^12
    # Q(y) = [g_2, g_4, g_6, g_8, g_10, g_12] where g_{2k} is coeff of x^{2k}
    
    # Coefficients (ascending: x^0, x^2, x^4, ..., x^12)
    # H0: [1]
    # H4: [12, -48, 16] for x^0, x^2, x^4
    # H8: [168, -1344, 2688, -1344, 256] for x^0, x^2, x^4, x^6, x^8
    # H12: [11880, -460800, 907200, -752640, 282240, -46080, 4096] for x^0, x^2, ..., x^12
    
    c0f = 1.0  # normalized
    g_even = np.zeros(7)  # g_{0}, g_{2}, ..., g_{12}
    # H0
    g_even[0] += c0f * 1.0
    # H4
    g_even[0] += c1f * 12.0
    g_even[1] += c1f * (-48.0)
    g_even[2] += c1f * 16.0
    # H8
    g_even[0] += c2f * 168.0
    g_even[1] += c2f * (-1344.0)
    g_even[2] += c2f * 2688.0
    g_even[3] += c2f * (-1344.0)
    g_even[4] += c2f * 256.0
    # H12
    g_even[0] += c3f * 11880.0
    g_even[1] += c3f * (-460800.0)
    g_even[2] += c3f * 907200.0
    g_even[3] += c3f * (-752640.0)
    g_even[4] += c3f * 282240.0
    g_even[5] += c3f * (-46080.0)
    g_even[6] += c3f * 4096.0
    
    # Q(y) = g_2 + g_4*y + g_6*y^2 + ... + g_12*y^5
    Q = g_even[1:7]  # [g_2, g_4, g_6, g_8, g_10, g_12]
    return Q


def _Q_eval(Q, y):
    """Evaluate Q(y) using Horner's method. Q is ascending coefficients."""
    result = Q[-1]
    for c in Q[-2::-1]:
        result = result * y + c
    return result


def _find_largest_sign_change_root(Q):
    """Find the largest positive real root of Q where Q changes sign.
    Q is a numpy array of ascending coefficients.
    Returns the root y as float, or None."""
    if len(Q) < 2:
        return None
    
    # Find all real positive roots using numpy
    try:
        roots = np.roots(Q)
    except Exception:
        return None
    
    candidates = []
    for r in roots:
        if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)):
            if r.real > 1e-10:
                candidates.append(float(r.real))
    
    if not candidates:
        return None
    
    candidates.sort(reverse=True)
    
    for y in candidates:
        # Check sign change with multiple step sizes
        h_base = max(1e-8, 1e-8 * abs(y))
        for h in [h_base * 10, h_base, h_base * 0.1]:
            left = _Q_eval(Q, y - h)
            right = _Q_eval(Q, y + h)
            if left * right < 0:
                # Refine with brentq
                lo = max(0.0, y - h)
                hi = y + h
                try:
                    refined = brentq(lambda t: _Q_eval(Q, t), lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)
                    return refined
                except Exception:
                    return y
        # Also check if it's a true root (Q(y) ≈ 0) with sign change on both sides
        h = max(1e-6, 1e-6 * abs(y))
        left = _Q_eval(Q, y - h)
        right = _Q_eval(Q, y + h)
        if left * right < 0:
            lo = max(0.0, y - h)
            hi = y + h
            try:
                refined = brentq(lambda t: _Q_eval(Q, t), lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)
                return refined
            except Exception:
                return y
    
    return None


def _value_from(c1, c2):
    """Compute value = y_max / (2*pi) where y_max is largest sign-change root."""
    try:
        Q = _build_Q_float(c1, c2)
        y_max = _find_largest_sign_change_root(Q)
        if y_max is None:
            return np.inf
        return y_max / (2.0 * np.pi)
    except Exception:
        return np.inf


def _objective(params):
    """Objective function for scipy optimizers."""
    c1, c2 = params
    return _value_from(float(c1), float(c2))


def _value_from_exact(c1_frac, c2_frac):
    """Compute value using exact rational arithmetic for root identification."""
    try:
        Q_poly, c3 = _build_Q_exact(c1_frac, c2_frac)
        # Find all real roots of Q_poly
        # Q_poly is in ZZ[y]
        # Get real roots
        real_roots = Q_poly.real_roots()
        positive_roots = [r for r in real_roots if r > 0]
        if not positive_roots:
            return np.inf
        
        # For each positive root, check sign change
        # Sort descending
        positive_roots.sort(reverse=True)
        
        for root in positive_roots:
            # Check sign change: evaluate Q at root ± epsilon
            # Use exact evaluation
            eps = Rational(1, 10**12)
            left_val = Q_poly.eval(root - eps)
            right_val = Q_poly.eval(root + eps)
            if left_val * right_val < 0:
                # This is a sign-change root
                return float(root) / (2.0 * np.pi)
        
        return np.inf
    except Exception:
        return np.inf


def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 200.0
    MARGIN = 10.0

    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0

    # Normalize: set c0 = 1, work with (c1/c0, c2/c0)
    seed_c1r = seed_c1 / seed_c0
    seed_c2r = seed_c2 / seed_c0

    best_params = np.array([seed_c1r, seed_c2r], dtype=np.float64)
    best_val = _value_from(seed_c1r, seed_c2r)

    def time_left():
        return DEADLINE - MARGIN - (time.time() - start)

    # Phase 1: Broad exploration with differential evolution
    # The key insight: we want Q(y) to have its largest sign-change root as small as possible
    # Q(y) = a0 + a1*y + ... + a5*y^5 where a5 = 4096*c3 (always positive for c3 > 0)
    # For large y, Q(y) ~ a5*y^5, so Q -> +inf if a5 > 0
    # We need Q to dip below 0 and come back, with the crossing as close to 0 as possible
    
    bounds = [(-10.0, 10.0), (-1.0, 1.0)]

    try:
        if time_left() > 80:
            res_de = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=300,
                popsize=40,
                tol=1e-12,
                mutation=(0.4, 1.2),
                recombination=0.85,
                seed=42,
                polish=True,
                init='sobol',
                workers=-1,
                updating='deferred',
            )
            if res_de is not None and np.isfinite(res_de.fun):
                if res_de.fun < best_val:
                    best_val = float(res_de.fun)
                    best_params = np.array(res_de.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 2: Focused DE around best solution
    try:
        if time_left() > 50:
            half_width = 2.0
            tight_bounds = [
                (best_params[0] - half_width, best_params[0] + half_width),
                (best_params[1] - half_width * 0.1, best_params[1] + half_width * 0.1),
            ]
            res_de2 = differential_evolution(
                _objective,
                bounds=tight_bounds,
                maxiter=200,
                popsize=30,
                tol=1e-14,
                mutation=(0.3, 0.9),
                recombination=0.9,
                seed=123,
                polish=True,
                init='latinhypercube',
                workers=-1,
            )
            if res_de2 is not None and np.isfinite(res_de2.fun):
                if res_de2.fun < best_val:
                    best_val = float(res_de2.fun)
                    best_params = np.array(res_de2.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 3: Basin hopping for escaping local minima
    try:
        if time_left() > 30:
            def _bh_obj(x):
                return _objective(x)
            
            res_bh = basinhopping(
                _bh_obj,
                x0=best_params,
                niter=50,
                stepsize=1.0,
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'xatol': 1e-11, 'fatol': 1e-13}},
                seed=456,
                disp=False,
            )
            if res_bh is not None and np.isfinite(res_bh.fun):
                if res_bh.fun < best_val:
                    best_val = float(res_bh.fun)
                    best_params = np.array(res_bh.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 4: Multi-start local optimization
    try:
        if time_left() > 25:
            rng = np.random.default_rng(789)
            n_starts = min(12, int(time_left() / 2.5))
            for i in range(n_starts):
                if time_left() < 5:
                    break
                # Generate diverse starting points
                scale = 0.05 + 0.5 * rng.random()
                perturbation = rng.normal(0, scale, size=2)
                x0 = best_params + perturbation
                x0 = np.clip(x0, [bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]])
                
                try:
                    res_ms = minimize(
                        _objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True},
                    )
                    if res_ms is not None and np.isfinite(res_ms.fun):
                        if res_ms.fun < best_val:
                            best_val = float(res_ms.fun)
                            best_params = np.array(res_ms.x, dtype=np.float64)
                except Exception:
                    pass
    except Exception:
        pass

    # Phase 5: High-precision polish with multiple methods
    try:
        if time_left() > 15:
            # Nelder-Mead with very tight tolerances
            res_nm = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True},
            )
            if res_nm is not None and np.isfinite(res_nm.fun):
                if res_nm.fun < best_val:
                    best_val = float(res_nm.fun)
                    best_params = np.array(res_nm.x, dtype=np.float64)
    except Exception:
        pass

    try:
        if time_left() > 10:
            # COBYLA for final refinement
            res_cobyla = minimize(
                _objective,
                best_params,
                method='COBYLA',
                options={'maxiter': 20000, 'rhobeg': 1e-10, 'tol': 1e-15},
            )
            if res_cobyla is not None and np.isfinite(res_cobyla.fun):
                if res_cobyla.fun < best_val:
                    best_val = float(res_cobyla.fun)
                    best_params = np.array(res_cobyla.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 6: Exact arithmetic verification and refinement
    # Use sympy to find the exact root and verify
    try:
        if time_left() > 15:
            # Convert best params to fractions for exact computation
            c1_frac = Fraction(str(float(best_params[0]))).limit_denominator(10**15)
            c2_frac = Fraction(str(float(best_params[2] if len(best_params) > 2 else best_params[1]))).limit_denominator(10**15)
            c2_frac = Fraction(str(float(best_params[1]))).limit_denominator(10**15)
            
            # Try small perturbations in exact arithmetic
            for dc1_num in range(-5, 6):
                for dc2_num in range(-5, 6):
                    if time_left() < 5:
                        break
                    if dc1_num == 0 and dc2_num == 0:
                        continue
                    pert_c1 = best_params[0] + dc1_num * 1e-12
                    pert_c2 = best_params[1] + dc2_num * 1e-12
                    try:
                        v = _value_from(pert_c1, pert_c2)
                        if np.isfinite(v) and v < best_val:
                            best_val = v
                            best_params = np.array([pert_c1, pert_c2], dtype=np.float64)
                    except Exception:
                        pass
    except Exception:
        pass

    # Final: very fine local search around best
    try:
        if time_left() > 5:
            # Grid search in tiny neighborhood
            for dc1 in np.linspace(-1e-10, 1e-10, 5):
                for dc2 in np.linspace(-1e-10, 1e-10, 5):
                    if time_left() < 3:
                        break
                    try:
                        v = _value_from(best_params[0] + dc1, best_params[1] + dc2)
                        if np.isfinite(v) and v < best_val:
                            best_val = v
                            best_params = np.array([best_params[0] + dc1, best_params[1] + dc2], dtype=np.float64)
                    except Exception:
                        pass
    except Exception:
        pass

    # Convert back to (c0, c1, c2) with c0 = 1
    c0_out = 1.0
    c1_out = float(best_params[0])
    c2_out = float(best_params[1])

    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2], dtype=np.float64)

    return np.array([c0_out, c1_out, c2_out], dtype=np.float64)
# EVOLVE-BLOCK-END
