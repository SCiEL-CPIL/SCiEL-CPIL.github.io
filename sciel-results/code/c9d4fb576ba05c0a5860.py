"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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


def _compute_integrals(a, b, c):
    """Compute I0, I2, J0 using precomputed Gauss-Hermite nodes.
    Returns (I0, I2, J0) and also P, P_prime for gradient computation."""
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    return I0, I2, J0, P, P_prime, d


def _objective_and_grad(params):
    """Compute the uncertainty product and its gradient with respect to (a, b, c)."""
    a, b, c = params
    I0, I2, J0, P, P_prime, d = _compute_integrals(a, b, c)
    
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    
    # Product = sqrt(I2 * J0) / I0
    # Let f = I2 * J0 / I0^2, then product = sqrt(f)
    # df/da = (dI2/da * J0 + I2 * dJ0/da) / I0^2 - 2 * I2 * J0 * dI0/da / I0^3
    # d(sqrt(f))/da = (1/(2*sqrt(f))) * df/da
    
    # Compute derivatives of I0, I2, J0 with respect to a, b, c
    # P = a*H0 + b*H4 + c*H8 + d*H12, where d = -(a+3b+105c)/10395
    # dP/da = H0 + H12 * dd/da = H0 - H12/10395
    # dP/db = H4 + H12 * dd/db = H4 - 3*H12/10395
    # dP/dc = H8 + H12 * dd/dc = H8 - 105*H12/10395
    
    dP_da = _H0 - _H12 / 10395.0
    dP_db = _H4 - 3 * _H12 / 10395.0
    dP_dc = _H8 - 105 * _H12 / 10395.0
    
    # dI0/da = 2 * sum(w * P * dP_da)
    dI0_da = 2 * np.dot(_W_I0, P * dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * dP_dc)
    
    # dI2/da = 2 * sum(w*x^2 * P * dP_da)
    dI2_da = 2 * np.dot(_W_I2, P * dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * dP_dc)
    
    # For J0: g_prime_poly = P_prime - x*P
    # dP_prime/da = DH12 * dd/da = -DH12/10395  (since d=-(a+3b+105c)/10395, dd/da = -1/10395)
    # Wait: P_prime = b*DH4 + c*DH8 + d*DH12
    # dP_prime/da = DH12 * dd/da = -DH12/10395
    # dP_prime/db = DH4 + DH12 * dd/db = DH4 - 3*DH12/10395
    # dP_prime/dc = DH8 + DH12 * dd/dc = DH8 - 105*DH12/10395
    
    dPp_da = -_DH12 / 10395.0
    dPp_db = _DH4 - 3 * _DH12 / 10395.0
    dPp_dc = _DH8 - 105 * _DH12 / 10395.0
    
    g_pp = P_prime - _X_GQ * P  # current g_prime_poly
    
    # d(g_pp)/da = dPp_da - x * dP_da
    dgp_da = dPp_da - _X_GQ * dP_da
    dgp_db = dPp_db - _X_GQ * dP_db
    dgp_dc = dPp_dc - _X_GQ * dP_dc
    
    dJ0_da = 2 * np.dot(_W_I0, g_pp * dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_pp * dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_pp * dgp_dc)
    
    # f = I2 * J0 / I0^2
    # df/da = (dI2_da * J0 + I2 * dJ0_da) / I0^2 - 2 * I2 * J0 * dI0_da / I0^3
    # d(sqrt(f))/da = (1/(2*sqrt(f))) * df/da
    # = (dI2_da * J0 + I2 * dJ0_da) / (2 * I0^2 * sqrt(f)) - I2 * J0 * dI0_da / (I0^3 * sqrt(f))
    # = (dI2_da * J0 + I2 * dJ0_da) / (2 * I0 * I2 * J0) - I2 * J0 * dI0_da / (I0^2 * I2 * J0)
    # Simplify: sqrt(f) = sqrt(I2*J0)/I0, so 2*sqrt(f) = 2*sqrt(I2*J0)/I0
    # d(sqrt(f))/da = (dI2_da * J0 + I2 * dJ0_da) / (2 * I2 * J0) - dI0_da / I0
    
    product = np.sqrt(I2 * J0) / I0
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0
    
    return product, np.array([grad_a, grad_b, grad_c])


def construct():
    start_time = time.time()
    deadline = 200.0  # seconds (safety margin before 210s kill)
    
    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)
    
    # Define bounds for global search
    # Based on seed and typical scales
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-1.0, 1.0)]
    
    # 1. Global Search using Differential Evolution
    # DE is robust to local minima and doesn't need gradients
    try:
        if time.time() < deadline - 60:
            # Use a smaller population and fewer iterations to fit in time
            # DE can be slow, so we limit it
            de_result = minimize(
                lambda x: _objective_and_grad(x)[0],
                seed_result, # x0 is not used by DE directly in same way, but bounds are
                method='differential_evolution',
                bounds=bounds,
                options={
                    'popsize': 10,
                    'maxiter': 200,
                    'tol': 1e-10,
                    'mutation': (0.5, 1.0),
                    'recombination': 0.7,
                    'polish': False, # We will polish manually
                    'init': 'sobol'
                }
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
    except Exception:
        pass

    # 2. Local Polish using L-BFGS-B from best global candidate
    if time.time() < deadline - 15:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # 3. Final high-precision polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if result.fun < best_score:
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
