"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


# Precompute Gauss-Hermite nodes and weights (20 points is exact for degree <= 39)
_N_GQ = 20
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ

# Hermite polynomials (physicists' convention) evaluated at nodes
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 84x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 100800x^2 + 10395
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

# Derivatives
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11
_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Quadrature weights
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2


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
    """Compute the uncertainty product and its gradient with respect to (a, b, c)."""
    a, b, c = params
    I0, I2, J0, P, P_prime, d = _compute_integrals(a, b, c)
    
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    
    # Precompute dP derivatives
    dP_da = _H0 - _H12 / 10395.0
    dP_db = _H4 - 3 * _H12 / 10395.0
    dP_dc = _H8 - 105 * _H12 / 10395.0
    
    dI0_da = 2 * np.dot(_W_I0, P * dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * dP_dc)
    
    dI2_da = 2 * np.dot(_W_I2, P * dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * dP_dc)
    
    dPp_da = -_DH12 / 10395.0
    dPp_db = _DH4 - 3 * _DH12 / 10395.0
    dPp_dc = _DH8 - 105 * _DH12 / 10395.0
    
    g_pp = P_prime - _X_GQ * P
    
    dgp_da = dPp_da - _X_GQ * dP_da
    dgp_db = dPp_db - _X_GQ * dP_db
    dgp_dc = dPp_dc - _X_GQ * dP_dc
    
    dJ0_da = 2 * np.dot(_W_I0, g_pp * dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_pp * dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_pp * dgp_dc)
    
    product = np.sqrt(I2 * J0) / I0
    grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    grad_b = (dI2_db * J0 + I2 * dJ0_db) / (2.0 * I2 * J0) - dI0_db / I0
    grad_c = (dI2_dc * J0 + I2 * dJ0_dc) / (2.0 * I2 * J0) - dI0_dc / I0
    
    return product, np.array([grad_a, grad_b, grad_c])


def construct():
    start_time = time.time()
    deadline = 195.0  # safety margin before 210s kill
    
    # Seed value
    seed_result = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    start_points = [best_params.copy()]
    
    # Perturbed versions of the seed
    for i in range(30):
        perturbation = rng.randn(3) * 0.05
        start_points.append(seed_result + perturbation)
    
    # Wider random points
    for i in range(30):
        start_points.append(rng.randn(3) * 0.3)
    
    # Structured grid points around seed
    for da in [-0.1, -0.05, 0, 0.05, 0.1]:
        for db in [-0.02, -0.01, 0, 0.01, 0.02]:
            for dc in [-0.001, 0, 0.001]:
                start_points.append(seed_result + np.array([da, db, dc]))
    
    # Optimize from each starting point using L-BFGS-B with analytic gradient
    for sp in start_points:
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                sp,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Refine with tighter tolerances
    if time.time() < deadline - 40:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Newton-like refinement using the Hessian-free approach:
    # Try a few steps of steepest descent with adaptive step size
    if time.time() < deadline - 10:
        current = best_params.copy()
        current_score, current_grad = _objective_and_grad(current)
        
        for step_iter in range(50):
            if time.time() > deadline - 5:
                break
            if np.linalg.norm(current_grad) < 1e-15:
                break
            
            # Adaptive step size
            step_size = 0.01
            for _ in range(20):
                new_params = current - step_size * current_grad
                new_score, new_grad = _objective_and_grad(new_params)
                if new_score < current_score:
                    current = new_params
                    current_score = new_score
                    current_grad = new_grad
                    break
                step_size *= 0.5
            
            if step_size < 1e-10:
                break
            
            if current_score < best_score:
                best_score = current_score
                best_params = current.copy()
    
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
