"""Search program for the Uncertainty Inequality task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite

_DEADLINE = 180.0
_SEED = np.array([-113.0 / 100.0, 0.5, 1.0 / 3240.0])
_PI = np.pi


def _q_coeffs(c0, c1, c2):
    """Return (c3, [q0..q5]) for q = g/x^2 with g = c0 H0 + c1 H4 + c2 H8 + c3 H12."""
    # H_n(0) for even n: H0=1, H4=-12, H8=1680, H12=-498960
    c3 = -(c0 * 1.0 + c1 * (-12.0) + c2 * 1680.0) / (-498960.0)
    # g = c0*1 + c1*(x^4 - 6x^2 + 3) + c2*(x^8 - 28x^6 + 210x^4 - 168)
    #     + c3*(x^12 - 66x^10 + 1155x^8 - 6930x^6 + 10395x^4 - 252)
    # q = g/x^2 (g has no x^0 or x^1 terms, so divides exactly)
    q0 = c1 * (-6.0) + c2 * 210.0 + c3 * (-6930.0)
    q1 = c1 * 1.0 + c2 * (-28.0) + c3 * 1155.0
    q2 = c2 * 1.0 + c3 * (-66.0)
    q3 = c3
    q4 = 0.0
    q5 = 0.0
    return c3, np.array([q0, q1, q2, q3, q4, q5])


def _largest_sign_changing_root(q):
    """Largest real root of q(x) at which q changes sign."""
    roots = np.roots(q)
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-10])
    best = None
    for r in real_roots:
        # check sign change: evaluate q at r-eps and r+eps
        eps = 1e-8 * (1.0 + abs(r))
        left = np.polyval(q, r - eps)
        right = np.polyval(q, r + eps)
        if left == 0.0:
            left = np.polyval(q, r - 10 * eps)
        if right == 0.0:
            right = np.polyval(q, r + 10 * eps)
        if left * right < 0:
            if best is None or r > best:
                best = r
    if best is None:
        return None
    return best


def _compute_value(c0, c1, c2):
    """Return value = r^2/(2*pi) or inf if no valid sign-changing root."""
    c3, q = _q_coeffs(c0, c1, c2)
    r = _largest_sign_changing_root(q)
    if r is None:
        return np.inf
    return r * r / (2.0 * _PI)


def _is_valid(c0, c1, c2):
    """Check g tends to positive limit at infinity (or can be flipped)."""
    c3, q = _q_coeffs(c0, c1, c2)
    # leading coefficient of g is c3 (from H12)
    # g tends to sign(c3) * inf. If negative, g is replaced by -g, so always valid
    # as long as q divides exactly (it does by construction) and has a sign-changing root
    r = _largest_sign_changing_root(q)
    return r is not None


def construct():
    t0 = time.time()
    best_val = _compute_value(_SEED[0], _SEED[1], _SEED[2])
    best_c = _SEED.copy()

    def _obj(x):
        v = _compute_value(x[0], x[1], x[2])
        if not np.isfinite(v):
            return 1e10
        return v

    # Local refinement around seed
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() - t0 > _DEADLINE - 30:
            break
        try:
            res = minimize(_obj, best_c, method=method,
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Random restarts in a bounded box around seed
    rng = np.random.default_rng(42)
    scale = np.array([0.5, 0.1, 0.001])
    for i in range(2000):
        if time.time() - t0 > _DEADLINE - 20:
            break
        cand = best_c + rng.normal(0, scale)
        v = _obj(cand)
        if np.isfinite(v) and v < best_val:
            best_val = v
            best_c = cand.copy()
            # local polish
            try:
                res = minimize(_obj, best_c, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-11, 'fatol': 1e-13})
                if res.success and np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass

    # Final check
    if time.time() - t0 > _DEADLINE - 5:
        return _SEED.copy()

    if not np.isfinite(best_val) or best_val >= _compute_value(_SEED[0], _SEED[1], _SEED[2]):
        return _SEED.copy()

    return best_c
# EVOLVE-BLOCK-END
