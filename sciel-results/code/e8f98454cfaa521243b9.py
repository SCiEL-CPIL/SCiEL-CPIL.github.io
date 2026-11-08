"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 210.0
_SAFETY_MARGIN = 5.0


def _precompute_constants():
    """Precompute all constants needed for the bound calculation."""
    # H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
    def h_at_0(n):
        if n % 2 != 0:
            return 0.0
        k = n // 2
        val = 1.0
        for i in range(1, n + 1):
            val *= i
        for i in range(1, k + 1):
            val /= i
        if k % 2 != 0:
            val = -val
        return val

    degrees = [0, 4, 8, 12]
    h0 = np.array([h_at_0(d) for d in degrees])
    
    # norm_sq coefficient: sqrt(pi) * 2^k * k!
    sqrt_pi = np.sqrt(np.pi)
    norm_coeff = np.zeros(4)
    for i, k in enumerate(degrees):
        fact = 1.0
        for j in range(1, k + 1):
            fact *= j
        norm_coeff[i] = sqrt_pi * (2.0 ** k) * fact
    
    # x^2 moment matrix: <H_l, x^2 H_k>
    # x^2 H_k = 0.25 H_{k+2} + 0.5k H_k + 0.25k(k-1) H_{k-2}
    x2_matrix = np.zeros((4, 4))
    for i, k in enumerate(degrees):
        for j, l in enumerate(degrees):
            val = 0.0
            if l == k + 2:
                fact_l = 1.0
                for jj in range(1, l + 1):
                    fact_l *= jj
                val += 0.25 * sqrt_pi * (2.0 ** l) * fact_l
            elif l == k:
                fact_l = 1.0
                for jj in range(1, l + 1):
                    fact_l *= jj
                val += 0.5 * k * sqrt_pi * (2.0 ** l) * fact_l
            elif l == k - 2 and k >= 2:
                fact_l = 1.0
                for jj in range(1, l + 1):
                    fact_l *= jj
                val += 0.25 * k * (k - 1) * sqrt_pi * (2.0 ** l) * fact_l
            x2_matrix[i, j] = val
    
    return h0, norm_coeff, x2_matrix


_H0, _NORM_COEFF, _X2_MATRIX = _precompute_constants()
_H12_0 = _H0[3]


def _compute_bound_and_grad(c0, c4, c8):
    """Compute the bound and its gradient w.r.t. (c0, c4, c8)."""
    c12 = -(c0 * _H0[0] + c4 * _H0[1] + c8 * _H0[2]) / _H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    # Norm squared
    norm_sq = np.sum(coeffs ** 2 * _NORM_COEFF)
    if norm_sq < 1e-15:
        return 1e10, np.zeros(3)
    
    # x^2 moment
    x2_moment = coeffs @ _X2_MATRIX @ coeffs
    
    A_x_sq = x2_moment / norm_sq
    bound = A_x_sq * A_x_sq  # (A_x)^4 since A(F) = A_x by Fourier invariance
    
    # Gradient
    # bound = (x2_moment / norm_sq)^2
    # d(bound)/d(coeffs) = 2 * (x2_moment/norm_sq) * (d(x2_moment)/d(coeffs)/norm_sq - x2_moment * d(norm_sq)/d(coeffs)/norm_sq^2)
    
    dx2 = 2.0 * _X2_MATRIX @ coeffs
    dnorm = 2.0 * coeffs * _NORM_COEFF
    
    # d(bound)/d(coeffs_i) for i in {0,1,2,3}
    dbound_dc = 2.0 * A_x_sq * (dx2 / norm_sq - x2_moment * dnorm / (norm_sq ** 2))
    
    # Chain rule for c12
    dc12_dc = np.array([-_H0[0] / _H12_0, -_H0[1] / _H12_0, -_H0[2] / _H12_0])
    
    grad = dbound_dc[:3] + dbound_dc[3] * dc12_dc
    
    return bound, grad


def _objective(coeffs):
    """Objective function for scipy minimize."""
    bound, _ = _compute_bound_and_grad(coeffs[0], coeffs[1], coeffs[2])
    return bound


def _objective_and_grad(coeffs):
    """Objective and gradient for scipy minimize."""
    bound, grad = _compute_bound_and_grad(coeffs[0], coeffs[1], coeffs[2])
    return bound, grad


def construct():
    best_score = 1e10
    best_coeffs = np.array([-1.13, 0.04, 0.000308642])
    
    # Initial guess from seed
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    init_score, _ = _compute_bound_and_grad(c0_init, c4_init, c8_init)
    best_score = init_score
    best_coeffs = np.array([c0_init, c4_init, c8_init])
    
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return best_coeffs
    
    # Multi-start BFGS with analytic gradients
    np.random.seed(42)
    
    # Generate diverse starting points
    starts = [np.array([c0_init, c4_init, c8_init])]
    
    # Add perturbed starts
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(5):
            perturbation = np.random.normal(0, scale, 3)
            starts.append(np.array([c0_init, c4_init, c8_init]) + perturbation)
    
    # Add some structured starts
    for c0_try in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        for c4_try in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            for c8_try in [-0.001, 0.0, 0.001]:
                starts.append(np.array([c0_try, c4_try, c8_try]))
    
    for start in starts:
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break
        
        try:
            result = minimize(_objective_and_grad, start, method='BFGS',
                            jac=True,
                            options={'maxiter': 500, 'gtol': 1e-10, 'factr': 1e-10})
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - _START_TIME < _DEADLINE - _SAFETY_MARGIN - 10:
        try:
            result = minimize(_objective_and_grad, best_coeffs, method='BFGS',
                            jac=True,
                            options={'maxiter': 1000, 'gtol': 1e-12, 'factr': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
