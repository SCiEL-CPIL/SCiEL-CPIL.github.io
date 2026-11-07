"""Structural approach: Global search via Differential Evolution on scaled coefficients + Newton polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import differential_evolution

_DEADLINE = 175.0
_t_start = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t_start)


def _q_coefficients(c0, c1, c2):
    """Return [b0..b5] where q(y) = b0 + b1*y + ... + b5*y^5, y=x^2."""
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
    b0 = -48.0 * c1 - 40320.0 * c2 - 5109120.0 * c3
    b1 = 16.0 * c1 + 26880.0 * c2 + 9072000.0 * c3
    b2 = -4608.0 * c2 - 6005760.0 * c3
    b3 = 256.0 * c2 + 1424640.0 * c3
    b4 = -142464.0 * c3
    b5 = 4096.0 * c3
    return np.array([b0, b1, b2, b3, b4, b5])


def _q_eval(coeffs, y):
    """Horner evaluation of q(y)."""
    r = 0.0
    for b in coeffs:
        r = r * y + b
    return r


def _q_deriv(coeffs, y):
    """Derivative q'(y) via Horner."""
    r = 0.0
    for i in range(len(coeffs) - 1, 0, -1):
        r = r * y + i * coeffs[i]
    return r


def _find_largest_sign_change_root(coeffs):
    """Find largest real positive root where q changes sign. Returns y or None."""
    poly_coeffs = np.array(coeffs[::-1])
    roots = np.roots(poly_coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-12:
            real_roots.append(r.real)
    if not real_roots:
        return None
    real_roots.sort(reverse=True)
    for y in real_roots:
        eps = max(1e-7, abs(y) * 1e-7)
        qb = _q_eval(coeffs, y - eps)
        qa = _q_eval(coeffs, y + eps)
        if qb * qa < 0:
            return y
    return None


def _evaluate_value(c0, c1, c2):
    """Compute value = r^2/(2*pi)."""
    coeffs = _q_coefficients(c0, c1, c2)
    y_root = _find_largest_sign_change_root(coeffs)
    if y_root is None:
        return np.inf
    return y_root / (2.0 * np.pi)


# Seed
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_seed_value = _evaluate_value(_SEED[0], _SEED[1], _SEED[2])


def _newton_double_root(c0, c1, c2, y_target, max_iter=50, tol=1e-14):
    """
    Newton's method to find (c1, c2) such that q(y_target)=0 and q'(y_target)=0,
    holding c0 fixed. Returns (c1, c2, success).
    """
    c1_cur = c1
    c2_cur = c2
    dc3_dc1 = -12.0 / 18480.0
    dc3_dc2 = -168.0 / 18480.0
    
    db0_dc1 = -48.0 + (-5109120.0) * dc3_dc1
    db1_dc1 = 16.0 + 9072000.0 * dc3_dc1
    db2_dc1 = 0.0 + (-6005760.0) * dc3_dc1
    db3_dc1 = 0.0 + 1424640.0 * dc3_dc1
    db4_dc1 = 0.0 + (-142464.0) * dc3_dc1
    db5_dc1 = 0.0 + 4096.0 * dc3_dc1
    
    db0_dc2 = -40320.0 + (-5109120.0) * dc3_dc2
    db1_dc2 = 26880.0 + 9072000.0 * dc3_dc2
    db2_dc2 = -4608.0 + (-6005760.0) * dc3_dc2
    db3_dc2 = 256.0 + 1424640.0 * dc3_dc2
    db4_dc2 = 0.0 + (-142464.0) * dc3_dc2
    db5_dc2 = 0.0 + 4096.0 * dc3_dc2
    
    y = y_target
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    y5 = y3 * y2
    
    df1_dc1 = db0_dc1 + db1_dc1 * y + db2_dc1 * y2 + db3_dc1 * y3 + db4_dc1 * y4 + db5_dc1 * y5
    df1_dc2 = db0_dc2 + db1_dc2 * y + db2_dc2 * y2 + db3_dc2 * y3 + db4_dc2 * y4 + db5_dc2 * y5
    df2_dc1 = db1_dc1 + 2.0 * db2_dc1 * y + 3.0 * db3_dc1 * y2 + 4.0 * db4_dc1 * y3 + 5.0 * db5_dc1 * y4
    df2_dc2 = db1_dc2 + 2.0 * db2_dc2 * y + 3.0 * db3_dc2 * y2 + 4.0 * db4_dc2 * y3 + 5.0 * db5_dc2 * y4
    
    J = np.array([[df1_dc1, df1_dc2],
                  [df2_dc1, df2_dc2]])
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if abs(det) < 1e-30:
        return c1_cur, c2_cur, False
    J_inv = np.array([[J[1, 1], -J[0, 1]],
                      [-J[1, 0], J[0, 0]]]) / det
    
    best_c1, best_c2, best_err = c1_cur, c2_cur, np.inf
    for _ in range(max_iter):
        coeffs = _q_coefficients(c0, c1_cur, c2_cur)
        f1 = _q_eval(coeffs, y_target)
        f2 = _q_deriv(coeffs, y_target)
        err = abs(f1) + abs(f2)
        if err < best_err:
            best_err = err
            best_c1, best_c2 = c1_cur, c2_cur
        if abs(f1) < tol and abs(f2) < tol:
            return c1_cur, c2_cur, True
        delta = J_inv @ np.array([f1, f2])
        step_size = 1.0
        for _ in range(10):
            c1_new = c1_cur - step_size * delta[0]
            c2_new = c2_cur - step_size * delta[1]
            coeffs_new = _q_coefficients(c0, c1_new, c2_new)
            f1_new = _q_eval(coeffs_new, y_target)
            f2_new = _q_deriv(coeffs_new, y_target)
            if abs(f1_new) + abs(f2_new) < err:
                c1_cur, c2_cur = c1_new, c2_new
                break
            step_size *= 0.5
        else:
            break
    coeffs = _q_coefficients(c0, best_c1, best_c2)
    f1 = _q_eval(coeffs, y_target)
    f2 = _q_deriv(coeffs, y_target)
    return best_c1, best_c2, (abs(f1) < 1e-10 and abs(f2) < 1e-10)


def construct():
    """
    Search for coefficients [c0, c1, c2] minimizing value = r^2/(2*pi).
    Uses scale invariance to fix c0=1 and Differential Evolution for global search.
    """
    global _t_start
    _t_start = time.time()

    # Scale seed to c0=1
    scale = 1.0 / _SEED[0]
    seed_scaled = _SEED * scale
    # seed_scaled is [1, c1_scaled, c2_scaled]
    best_params = _SEED.copy()
    best_value = _seed_value

    # Objective function for DE (minimizes value for c0=1)
    def de_objective(x):
        if _remaining() < 2:
            return 1e10
        c1, c2 = x
        val = _evaluate_value(1.0, c1, c2)
        if not np.isfinite(val):
            return 1e10
        return val

    # Bounds for c1, c2.
    # Based on seed and typical polynomial coefficient ranges.
    # We can expand these if needed, but DE is robust.
    bounds = [(-1.0, 1.0), (-0.1, 0.1)]
    
    # Run Differential Evolution
    try:
        if _remaining() > 30:
            res = differential_evolution(
                de_objective, bounds,
                maxiter=50, popsize=10, tol=1e-6,
                seed=42, polish=False,
                init='sobol'
            )
            if res.success or res.fun < best_value:
                c1_opt, c2_opt = res.x
                val_opt = _evaluate_value(1.0, c1_opt, c2_opt)
                if np.isfinite(val_opt) and val_opt < best_value:
                    best_value = val_opt
                    best_params = np.array([1.0, c1_opt, c2_opt])
    except Exception:
        pass

    # Local refinement using Newton's method from the best point found
    if _remaining() > 10 and best_value < np.inf:
        c0_best, c1_best, c2_best = best_params
        # Try to polish by finding a double root near the current largest sign change
        y_current = best_value * 2.0 * np.pi
        # Use Newton to find a double root at y_current (or slightly lower)
        # This often improves the value if the current root is not exactly a double root
        for y_t in [y_current * 0.99, y_current * 0.98, y_current * 0.95, y_current]:
            if _remaining() < 5:
                break
            c1_new, c2_new, success = _newton_double_root(
                c0_best, c1_best, c2_best, y_t, max_iter=50, tol=1e-12)
            if success:
                val_new = _evaluate_value(c0_best, c1_new, c2_new)
                if np.isfinite(val_new) and val_new < best_value:
                    best_value = val_new
                    best_params = np.array([c0_best, c1_new, c2_new])

    return np.array(best_params)
# EVOLVE-BLOCK-END
