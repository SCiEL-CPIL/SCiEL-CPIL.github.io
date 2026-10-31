"""Uncertainty inequality optimization with analytic gradient and multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


# Precompute Gauss-Hermite nodes and weights (15 points is exact for degree <= 29)
_N_GQ = 15
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395
# Precompute for derivative: P'(x) terms
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11
# Derivative of H4: 4x^3 - 12x
_DH4 = 4 * _X3 - 12 * _X_GQ
# Derivative of H8: 8x^7 - 168x^5 + 840x^3 - 840x
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
# Derivative of H12: 12x^11 - 840x^9 + 10080x^7 - 75600x^5 + 221760x^3 - 201600x
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Weights for integrals: w_i * 1, w_i * x_i^2
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2

# Precompute derivative vectors for analytic gradient
_dP_da = _H0 - _H12 / 10395.0
_dP_db = _H4 - 3 * _H12 / 10395.0
_dP_dc = _H8 - 105 * _H12 / 10395.0
_dPp_da = -_DH12 / 10395.0
_dPp_db = _DH4 - 3 * _DH12 / 10395.0
_dPp_dc = _DH8 - 105 * _DH12 / 10395.0
_dgp_da = _dPp_da - _X_GQ * _dP_da
_dgp_db = _dPp_db - _X_GQ * _dP_db
_dgp_dc = _dPp_dc - _X_GQ * _dP_dc


def _compute_integrals(a, b, c):
    """Compute I0, I2, J0 using precomputed Gauss-Hermite nodes."""
    d = -(a + 3 * b + 105 * c) / 10395.0
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    return I0, I2, J0, P, P_prime, d


def _objective_and_grad(params):
    """Compute the uncertainty product and its analytic gradient with respect to (a, b, c)."""
    a, b, c = params
    I0, I2, J0, P, P_prime, d = _compute_integrals(a, b, c)

    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])

    # Derivatives of I0, I2, J0 with respect to a, b, c
    dI0_da = 2 * np.dot(_W_I0, P * _dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * _dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * _dP_dc)

    dI2_da = 2 * np.dot(_W_I2, P * _dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * _dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * _dP_dc)

    g_pp = P_prime - _X_GQ * P

    dJ0_da = 2 * np.dot(_W_I0, g_pp * _dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_pp * _dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_pp * _dgp_dc)

    product = np.sqrt(I2 * J0) / I0
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0

    return product, np.array([grad_a, grad_b, grad_c])


def _objective_only(params):
    return _objective_and_grad(params)[0]


def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # 15s safety margin before 210s kill

    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)

    # Phase 1: Multi-start with diverse initial points
    rng = np.random.default_rng(42)
    starts = []

    # Around the seed
    for _ in range(15):
        s = seed_result + rng.normal(0, 0.1, 3)
        starts.append(s)

    # Wider exploration
    for _ in range(30):
        s = np.array([rng.uniform(-3.0, 2.0), rng.uniform(-0.5, 0.5), rng.uniform(-0.01, 0.01)])
        starts.append(s)

    # Structured grid
    for a0 in np.linspace(-3.0, 1.5, 8):
        for a4 in np.linspace(-0.3, 0.3, 5):
            a8 = 0.0005 * rng.normal()
            starts.append(np.array([a0, a4, a8]))

    # Evaluate all starts
    for s in starts:
        if time.time() > deadline - 80:
            break
        try:
            val, _ = _objective_and_grad(s)
            if val < best_score and np.isfinite(val):
                best_score = val
                best_params = s.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B from best starting point
    if time.time() < deadline - 40:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if result.fun < best_score and np.isfinite(result.fun):
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead high-precision polish
    if time.time() < deadline - 15:
        try:
            nm_res = minimize(
                _objective_only,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18}
            )
            if nm_res.fun < best_score and np.isfinite(nm_res.fun):
                best_score = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B polish from Nelder-Mead result
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_score and np.isfinite(result.fun):
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Final verification
    final_score, _ = _objective_and_grad(best_params)
    if final_score < best_score:
        best_score = final_score

    # Ensure we don't return worse than seed
    seed_score, _ = _objective_and_grad(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()

    return best_params
# EVOLVE-BLOCK-END
