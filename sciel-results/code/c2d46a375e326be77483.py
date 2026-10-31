"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


# --- Precompute Gauss-Hermite quadrature (15 nodes exact for degree <= 29) ---
_N_GQ = 15
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11
_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2

# Direction (scale-invariant) derivative basis, precomputed once.
# d = -(a + 3b + 105c)/10395
_D12 = 1.0 / 10395.0
_dP_da = _H0 - _D12 * _H12
_dP_db = _H4 - 3 * _D12 * _H12
_dP_dc = _H8 - 105 * _D12 * _H12
_dPp_da = -_D12 * _DH12
_dPp_db = _DH4 - 3 * _D12 * _DH12
_dPp_dc = _DH8 - 105 * _D12 * _DH12
_dgp_da = _dPp_da - _X_GQ * _dP_da
_dgp_db = _dPp_db - _X_GQ * _dP_db
_dgp_dc = _dPp_dc - _X_GQ * _dP_dc


def _compute_integrals(a, b, c):
    d = -(a + 3 * b + 105 * c) * _D12
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_pp = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_pp * g_pp)
    return I0, I2, J0, P, g_pp


def _objective_and_grad(params):
    """Uncertainty product and analytic gradient w.r.t. (a,b,c).

    The objective is scale-invariant, so we evaluate it on the normalized
    direction of the parameter vector (epigraph / change-of-variables trick
    that removes the scale degree of freedom)."""
    a, b, c = params
    nrm = np.sqrt(a * a + b * b + c * c)
    if nrm < 1e-14:
        return 1e10, np.array([0.0, 0.0, 0.0])
    a, b, c = a / nrm, b / nrm, c / nrm

    I0, I2, J0, P, g_pp = _compute_integrals(a, b, c)
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])

    dI0_da = 2 * np.dot(_W_I0, P * _dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * _dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * _dP_dc)
    dI2_da = 2 * np.dot(_W_I2, P * _dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * _dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * _dP_dc)
    dJ0_da = 2 * np.dot(_W_I0, g_pp * _dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_pp * _dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_pp * _dgp_dc)

    product = np.sqrt(I2 * J0) / I0
    # d(sqrt(f))/dparam = (dI2*J0 + I2*dJ0)/(2*I2*J0) - dI0/I0
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0

    # Pull the direction-space gradient back to the original coordinates.
    # For a scale-invariant f, df/da = (1/nrm) * grad_dir . (e_a - u u^T).
    u = np.array([a, b, c])
    g = np.array([grad_a, grad_b, grad_c])
    proj = g - np.dot(g, u) * u
    return product, proj / nrm


def _refine(x0, deadline):
    """Local L-BFGS-B refinement with analytic gradient; returns best params."""
    try:
        res = minimize(
            lambda x: _objective_and_grad(x)[0],
            x0,
            jac=lambda x: _objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-13,
                     'maxls': 40, 'maxfun': 800},
        )
        return res.x
    except Exception:
        return x0


def construct():
    start_time = time.time()
    deadline = 195.0  # safety margin before the 210s kill

    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)

    rng = np.random.RandomState(42)

    # Diverse starting points on the unit-sphere direction (scale removed).
    starts = [best_params.copy()]
    for i in range(16):
        starts.append(seed_result + rng.randn(3) * 0.1)
    for i in range(24):
        v = rng.randn(3)
        starts.append(v / (np.linalg.norm(v) + 1e-15))
    for a_val in [-2, -1, -0.5, -0.2, 0.2, 0.5, 1]:
        for b_val in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]:
            v = np.array([a_val, b_val, 0.01])
            starts.append(v / (np.linalg.norm(v) + 1e-15))

    # Multi-start local refinement.
    for sp in starts:
        if time.time() - start_time > deadline - 25:
            break
        x = _refine(sp, deadline)
        val, _ = _objective_and_grad(x)
        if val < best_score:
            best_score = val
            best_params = x.copy()

    # High-precision final polish: nested tightening + tiny random restarts.
    if time.time() - start_time < deadline - 15:
        for _ in range(3):
            if time.time() - start_time > deadline - 15:
                break
            x = _refine(best_params, deadline)
            val, _ = _objective_and_grad(x)
            if val < best_score:
                best_score = val
                best_params = x.copy()
            # tiny perturbations to escape any near-flat basin
            for _ in range(6):
                if time.time() - start_time > deadline - 15:
                    break
                xp = best_params + rng.randn(3) * 1e-4
                x = _refine(xp, deadline)
                val, _ = _objective_and_grad(x)
                if val < best_score:
                    best_score = val
                    best_params = x.copy()

    # Verify and never return worse than the seed.
    final_score, _ = _objective_and_grad(best_params)
    if final_score < best_score:
        best_score = final_score
    seed_score, _ = _objective_and_grad(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()

    return best_params
# EVOLVE-BLOCK-END
