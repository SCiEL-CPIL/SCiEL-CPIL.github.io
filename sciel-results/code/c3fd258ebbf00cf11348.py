import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn]."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]
        prev2 = H[k - 2]
        curr = [0.0] * (k + 1)
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def _build_g(c0, c1, c2):
    """Build g coefficients and return them (with sign fix)."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = [0.0] * 13
    for i in range(len(_H0)):
        g[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g[i] += c3 * _H12[i]
    if g[12] < 0:
        g = [-c for c in g]
    return g

def _get_q_coeffs(g):
    """Return q = g/x^2 coefficients, or None if not divisible."""
    if abs(g[0]) > 1e-14 or abs(g[1]) > 1e-14:
        return None
    return g[2:]

def _find_largest_sign_changing_root(q_coeffs):
    """Find largest positive real root where q changes sign."""
    if q_coeffs is None or len(q_coeffs) == 0:
        return None
    coeffs_desc = list(reversed(q_coeffs))
    while len(coeffs_desc) > 1 and abs(coeffs_desc[0]) < 1e-15:
        coeffs_desc = coeffs_desc[1:]
    if len(coeffs_desc) == 0:
        return None
    try:
        roots = np.roots(coeffs_desc)
    except:
        return None
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if not real_roots:
        return None
    poly = np.poly1d(coeffs_desc)
    largest = None
    for r in real_roots:
        eps = 1e-7 * max(1.0, abs(r))
        val_left = poly(r - eps)
        val_right = poly(r + eps)
        if val_left * val_right < 0:
            if largest is None or r > largest:
                largest = r
    return largest

def _compute_value(c0, c1, c2):
    """Compute the exact value for given coefficients."""
    g = _build_g(c0, c1, c2)
    q = _get_q_coeffs(g)
    if q is None:
        return float('inf')
    r = _find_largest_sign_changing_root(q)
    if r is None:
        return float('inf')
    return r ** 2 / (2 * np.pi)

def _objective(x):
    """Objective function for optimization."""
    try:
        return _compute_value(x[0], x[1], x[2])
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s

    def elapsed():
        return time.time() - start_time

    # Seed
    seed_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best_c = seed_c.copy()
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])

    # Phase 1: Differential evolution for broad exploration
    if elapsed() < deadline - 80:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(
                _objective, bounds,
                maxiter=100, popsize=30, tol=1e-8, seed=42, polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except:
            pass

    # Phase 2: Multi-start local optimization with different strategies
    np.random.seed(123)
    num_starts = 50
    for i in range(num_starts):
        if elapsed() > deadline - 40:
            break

        if i == 0:
            x0 = best_c.copy()
        elif i % 5 == 1:
            x0 = best_c + np.random.normal(0, 0.5, size=3)
        elif i % 5 == 2:
            x0 = best_c + np.random.normal(0, 0.1, size=3)
        elif i % 5 == 3:
            x0 = np.random.uniform(-3, 3, size=3)
        elif i % 5 == 4:
            x0 = best_c + np.random.normal(0, 0.01, size=3)
        else:
            x0 = best_c + np.random.normal(0, 0.2, size=3)

        # Try Nelder-Mead
        try:
            result = minimize(
                _objective, x0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass

        # Try Powell
        if elapsed() < deadline - 40:
            try:
                result = minimize(
                    _objective, best_c, method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_c = result.x.copy()
            except:
                pass

    # Phase 3: Fine-tuning with very small steps
    if elapsed() < deadline - 15:
        for scale in [1e-3, 1e-4, 1e-5, 1e-6]:
            if elapsed() > deadline - 15:
                break
            try:
                result = minimize(
                    _objective, best_c, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': scale * 1e-3, 'fatol': scale * 1e-3, 'adaptive': True}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_c = result.x.copy()
            except:
                pass

    # Phase 4: Random perturbations and refinement
    if elapsed() < deadline - 10:
        np.random.seed(999)
        for _ in range(20):
            if elapsed() > deadline - 10:
                break
            perturbed = best_c + np.random.normal(0, 0.01, size=3)
            try:
                result = minimize(
                    _objective, perturbed, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_c = result.x.copy()
            except:
                pass

    # Final verification
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if final_val < best_val:
        best_val = final_val

    return best_c
