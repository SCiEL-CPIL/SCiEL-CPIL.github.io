"""Optimized search for coefficients c0, c1, c2 minimizing r^2/(2*pi)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Deadline guard: evaluator kills after 210s, keep 30s safety margin
_DEADLINE = 180.0
_t_start = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t_start)


def _q_coefficients(c0, c1, c2):
    """
    Return coefficients [b1, b2, b3, b4, b5, b6] of q(y) where y = x^2.
    q(y) = g(x)/x^2 with g = c0*H0 + c1*H4 + c2*H8 + c3*H12.
    
    Physicists' Hermite polynomials:
      H0(x)  = 1
      H4(x)  = 16x^4 - 48x^2 + 12
      H8(x)  = 256x^8 - 4608x^6 + 26880x^4 - 40320x^2 + 168
      H12(x) = 4096x^12 - 142464x^10 + 1424640x^8 - 6005760x^6
               + 9072000x^4 - 5109120x^2 + 18480
    
    g(0) = c0 + 12*c1 + 168*c2 + 18480*c3 = 0
    => c3 = -(c0 + 12*c1 + 168*c2) / 18480
    
    g(x) = sum over even powers x^(2k), k=0..6
    q(y) = g(sqrt(y))/y = sum b_k * y^(k-1) for k=1..6, i.e., q is degree 5 in y.
    """
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0

    # Coefficients of x^(2k) in g for k=0..6:
    # H0:  [1]
    # H4:  [12, -48, 16]          (x^0, x^2, x^4)
    # H8:  [168, -40320, 26880, -4608, 256]  (x^0, x^2, x^4, x^6, x^8)
    # H12: [18480, -5109120, 9072000, -6005760, 1424640, -142464, 4096]
    #       (x^0, x^2, x^4, x^6, x^8, x^10, x^12)

    # g(x) coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12:
    g0 = c0 + 12.0 * c1 + 168.0 * c2 + 18480.0 * c3  # should be 0
    g1 = -48.0 * c1 - 40320.0 * c2 - 5109120.0 * c3
    g2 = 16.0 * c1 + 26880.0 * c2 + 9072000.0 * c3
    g3 = -4608.0 * c2 - 6005760.0 * c3
    g4 = 256.0 * c2 + 1424640.0 * c3
    g5 = -142464.0 * c3
    g6 = 4096.0 * c3

    # q(y) = g(x)/x^2 where y = x^2, so g(x) = g0 + g1*y + g2*y^2 + ... + g6*y^6
    # q(y) = (g1 + g2*y + g3*y^2 + g4*y^3 + g5*y^4 + g6*y^5) / 1
    # i.e., q is a polynomial in y of degree 5 with coefficients:
    b1 = g1  # y^0
    b2 = g2  # y^1
    b3 = g3  # y^2
    b4 = g4  # y^3
    b5 = g5  # y^4
    b6 = g6  # y^5

    return np.array([b1, b2, b3, b4, b5, b6])


def _find_largest_sign_change_root(coeffs):
    """
    Find the largest real positive root of q(y) where q changes sign.
    coeffs = [b1, b2, ..., b6] for polynomial b1 + b2*y + ... + b6*y^5.
    Returns r^2 = y_root (the largest such y), or None if no valid root.
    """
    # Use numpy roots for the polynomial (highest power first)
    poly_coeffs = np.array(coeffs[::-1])  # highest degree first
    roots = np.roots(poly_coeffs)

    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            real_roots.append(r.real)

    if not real_roots:
        return None

    real_roots.sort()

    # Check sign change at each root
    # Evaluate q(y) = b1 + b2*y + ... + b6*y^5
    def q_eval(y):
        result = 0.0
        power = 1.0
        for b in coeffs:
            result += b * power
            power *= y
        return result

    best_y = None
    for y in real_roots:
        # Check sign change: evaluate just below and just above
        eps = max(1e-8, abs(y) * 1e-8)
        q_below = q_eval(y - eps)
        q_above = q_eval(y + eps)
        if q_below * q_above < 0:
            if best_y is None or y > best_y:
                best_y = y

    return best_y


def _evaluate_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) for given coefficients."""
    coeffs = _q_coefficients(c0, c1, c2)
    y_root = _find_largest_sign_change_root(coeffs)
    if y_root is None:
        return np.inf
    return y_root / (2.0 * np.pi)


def _objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    val = _evaluate_value(c0, c1, c2)
    if not np.isfinite(val):
        return 1e10
    return val


# Seed coefficients
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_seed_value = _evaluate_value(_SEED[0], _SEED[1], _SEED[2])


def construct():
    """
    Search for coefficients [c0, c1, c2] minimizing value = r^2/(2*pi).
    Returns the best coefficients found.
    """
    global _t_start
    _t_start = time.time()

    best_params = _SEED.copy()
    best_value = _seed_value

    # Phase 1: Local optimization from seed using multiple methods
    if _remaining() > 10:
        # Try L-BFGS-B
        try:
            res = minimize(_objective, _SEED, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.success or res.fun < best_value:
                val = _evaluate_value(res.x[0], res.x[1], res.x[2])
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Try multiple random restarts with local optimization
    if _remaining() > 20:
        np.random.seed(42)
        n_restarts = 0
        max_restarts = 50

        for i in range(max_restarts):
            if _remaining() < 5:
                break
            n_restarts += 1
            # Random starting point in a reasonable range
            x0 = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-1, 1),
                np.random.uniform(-0.5, 0.5)
            ])
            try:
                res = minimize(_objective, x0, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
                if res.fun < best_value:
                    val = _evaluate_value(res.x[0], res.x[1], res.x[2])
                    if np.isfinite(val) and val < best_value:
                        best_value = val
                        best_params = res.x.copy()
            except Exception:
                pass

    # Phase 3: Differential evolution for global search if time permits
    if _remaining() > 30:
        try:
            bounds = [(-10, 10), (-2, 2), (-1, 1)]
            de_res = differential_evolution(_objective, bounds, maxiter=100,
                                            tol=1e-12, seed=42, polish=True,
                                            init='sobol', popsize=15,
                                            mutation=(0.3, 1.0), recombination=0.7)
            if de_res.fun < best_value:
                val = _evaluate_value(de_res.x[0], de_res.x[1], de_res.x[2])
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = de_res.x.copy()
        except Exception:
            pass

    # Phase 4: Final high-precision polish from best point
    if _remaining() > 5 and best_value < np.inf:
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_value:
                val = _evaluate_value(res.x[0], res.x[1], res.x[2])
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = res.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
