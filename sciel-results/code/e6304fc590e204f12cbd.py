"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _compute_bound(c0, c4, c8):
    """Compute the uncertainty bound C4 for given coefficients."""
    # H12 coefficient so that P(0) = 0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-945
    c12 = (c0 - 3.0 * c4 + 105.0 * c8) / 945.0

    degrees = [0, 4, 8, 12]
    coeffs = [c0, c4, c8, c12]

    # a_n = c_n * sqrt(2^n * n! * sqrt(pi))
    a = np.array([
        c * math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
        for n, c in zip(degrees, coeffs)
    ])

    norm_sq = np.sum(a**2)
    if norm_sq < 1e-30:
        return np.inf

    # x^2 is diagonal in this even-parity subspace (degrees differ by 4)
    x2_diag = np.array([(2.0 * n + 1.0) / 2.0 for n in degrees])
    x2_moment = np.sum(a**2 * x2_diag)

    ratio = x2_moment / norm_sq
    return ratio**2


def construct():
    """Optimize the three free coefficients to minimize the uncertainty bound."""
    t_start = time.time()
    deadline = t_start + 175.0  # Safety margin before 210s kill

    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_c = np.array([c0_init, c4_init, c8_init])
    best_val = _compute_bound(c0_init, c4_init, c8_init)

    def objective(c):
        return _compute_bound(c[0], c[1], c[2])

    # Multiple starting points for global search
    starts = [
        [c0_init, c4_init, c8_init],
        [0.0, 0.0, 0.0],
        [-1.0, 0.1, 0.01],
        [1.0, -0.1, -0.01],
        [-0.5, 0.05, 0.005],
        [0.5, -0.05, -0.005],
        [-2.0, 0.2, 0.02],
        [2.0, -0.2, -0.02],
        [-1.5, 0.15, 0.015],
        [1.5, -0.15, -0.015],
    ]

    for start in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(objective, np.array(start), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12,
                                    'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            continue

    # L-BFGS-B refinement
    if time.time() < deadline - 25.0:
        try:
            res = minimize(objective, best_c, method='L-BFGS-B',
                           options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Final tight polish
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-15,
                                    'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    return np.array(best_c)
# EVOLVE-BLOCK-END
