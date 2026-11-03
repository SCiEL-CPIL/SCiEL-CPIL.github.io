"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0

def _remaining():
    return _START_TIME + 210.0 - _DEADLINE_MARGIN - time.time()

def _hermite_coeffs(n):
    """Return dict mapping power -> coefficient for physicist's Hermite polynomial H_n(x)."""
    if n % 2 != 0:
        return {}
    coeffs = {}
    n2 = n // 2
    for k in range(n2 + 1):
        sign = 1 if k % 2 == 0 else -1
        coeff = sign * (2 ** (n - 2 * k)) * float(factorial(n)) / (float(factorial(k)) * float(factorial(n - 2 * k)))
        power = n - 2 * k
        coeffs[power] = coeffs.get(power, 0.0) + coeff
    return coeffs

def _combine_coeffs(a, b):
    """Multiply two polynomials represented as dicts {power: coeff}."""
    result = {}
    for p1, c1 in a.items():
        for p2, c2 in b.items():
            p = p1 + p2
            result[p] = result.get(p, 0.0) + c1 * c2
    return result

# Precompute Hermite coefficient dicts
_H0 = {0: 1.0}
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

# H_n(0) values
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _compute_upper_bound(c0, c4, c8):
    """Compute the upper bound on C4 = A(f)*A(F) for the Hermite-Gaussian test function."""
    c12 = -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0

    # Build polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    P = {}
    for c, hc in [(c0, _H0), (c4, _H4), (c8, _H8), (c12, _H12)]:
        for p, coeff in hc.items():
            P[p] = P.get(p, 0.0) + c * coeff

    # Compute ||g||_2^2 = ∫ P(x)^2 e^{-x^2} dx
    P2 = _combine_coeffs(P, P)

    norm_sq = 0.0
    for power, coeff in P2.items():
        if power % 2 != 0:
            continue
        n = power
        if n == 0:
            integral = np.sqrt(np.pi)
        else:
            double_fact = 1.0
            for k in range(1, n, 2):
                double_fact *= k
            integral = np.sqrt(np.pi) * double_fact / (2 ** (n / 2.0))
        norm_sq += coeff * integral

    if norm_sq <= 0:
        return float('inf')

    # Compute ||F(g)||_2^2 = 2π ∫ |P(iξ)|^2 e^{-ξ^2} dξ
    abs2_poly = {}
    for k, ak in P.items():
        for j, aj in P.items():
            power = k + j
            m = k + j
            mod4 = m % 4
            if mod4 == 0:
                i_pow = 1.0 + 0.0j
            elif mod4 == 1:
                i_pow = 0.0 + 1.0j
            elif mod4 == 2:
                i_pow = -1.0 + 0.0j
            else:
                i_pow = 0.0 - 1.0j

            coeff = ak * aj * ((-1) ** j) * i_pow
            abs2_poly[power] = abs2_poly.get(power, 0.0 + 0.0j) + coeff

    norm_f_sq = 0.0
    for power, coeff in abs2_poly.items():
        if power % 2 != 0:
            continue
        n = power
        c = coeff.real
        if n == 0:
            integral = np.sqrt(np.pi)
        else:
            double_fact = 1.0
            for k in range(1, n, 2):
                double_fact *= k
            integral = np.sqrt(np.pi) * double_fact / (2 ** (n / 2.0))
        norm_f_sq += c * integral

    norm_f_sq *= 2 * np.pi

    if norm_f_sq <= 0:
        return float('inf')

    return norm_sq * norm_f_sq

def _objective_2d(params):
    """Objective in reduced 2D space: params = [r4, r8] where c4 = r4*c0, c8 = r8*c0, c0=1."""
    r4, r8 = params
    return _compute_upper_bound(1.0, r4, r8)

def construct():
    # Seed values
    seed_c0, seed_c4, seed_c8 = -113.0/100, 1.0/25, 1.0/3240
    seed_val = _compute_upper_bound(seed_c0, seed_c4, seed_c8)

    # The objective is scale-invariant: scaling all c's by s scales the product by s^4.
    # So we fix c0 = 1 and optimize r4 = c4/c0 and r8 = c8/c0.
    # But we need to handle the sign of c0. Let's try both c0 > 0 and c0 < 0 cases.
    # Actually, since the product scales as s^4, and s^4 > 0 always, the sign of c0 doesn't matter
    # for the product value. But the constraint g(0)=0 involves c0, c4, c8, c12 linearly,
    # so flipping all signs flips c12 too, and the product is the same.
    # So we can fix c0 = 1 and optimize r4, r8.

    # Initial guess from seed (normalized to c0=1)
    if abs(seed_c0) > 1e-15:
        init_r4 = seed_c4 / seed_c0
        init_r8 = seed_c8 / seed_c0
    else:
        init_r4 = 0.0
        init_r8 = 0.0

    best_params_2d = np.array([init_r4, init_r8])
    best_val_2d = _objective_2d(best_params_2d)

    # Use differential evolution for global search in 2D
    if _remaining() > 30.0:
        try:
            bounds = [(-10.0, 10.0), (-10.0, 10.0)]
            result_de = differential_evolution(
                _objective_2d,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            if result_de.fun < best_val_2d:
                best_params_2d = result_de.x
                best_val_2d = result_de.fun
        except Exception:
            pass

    # Local refinement with Nelder-Mead
    if _remaining() > 10.0:
        try:
            result_nm = minimize(
                _objective_2d,
                best_params_2d,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000}
            )
            if result_nm.fun < best_val_2d:
                best_params_2d = result_nm.x
                best_val_2d = result_nm.fun
        except Exception:
            pass

    # Convert back to (c0, c4, c8) with c0 = 1
    best_c0 = 1.0
    best_c4 = best_params_2d[0]
    best_c8 = best_params_2d[1]

    # Also try with c0 = -1 to see if it gives a different (but equivalent) result
    # Since the product is scale-invariant, c0=-1 gives the same product.
    # But let's verify.

    # Return as [c0, c4, c8]
    return np.array([best_c0, best_c4, best_c8])
# EVOLVE-BLOCK-END
