import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    # Initial guess from seed program
    # Seed: [4.0, 8.0, ..., 48.0]
    x_seed = np.array([4.0 * (i + 1) for i in range(12)])
    z_init = np.log(x_seed)
    
    # Objective function to minimize: -log(Norm^2)
    # Maximizing Norm^2 minimizes the bound 1/Norm^2
    def objective(z):
        x = np.exp(z)
        # Sort roots to ensure stability in polynomial construction?
        # The product is commutative, but numerical stability of coefficient expansion
        # might benefit from sorted roots? Not strictly necessary for convolution.
        
        # Construct polynomial P(x) = prod (x - x_i)
        # Coefficients in descending power order: [1, -sum(x), sum(x_i x_j), ...]
        # It is easier to work with ascending powers for convolution with factorials.
        
        # Start with P = 1
        coeffs = np.array([1.0])
        for xi in x:
            # Multiply by (x - xi)
            # new_coeffs[k] = old_coeffs[k-1] - xi * old_coeffs[k]
            new_coeffs = np.zeros(len(coeffs) + 1)
            new_coeffs[1:] -= xi * coeffs
            new_coeffs[:-1] += coeffs
            coeffs = new_coeffs
            
        # coeffs[k] is coefficient of x^k in P(x)
        # P(x)^2 coefficients q_k
        # q_k = sum_{j=0}^k coeffs[j] * coeffs[k-j]
        norm_sq = 0.0
        for k in range(len(coeffs)):
            q_k = 0.0
            for j in range(k + 1):
                q_k += coeffs[j] * coeffs[k - j]
            # Integral of x^k e^-x is k!
            # Compute k!
            # To avoid overflow in factorial for large k, we can use log space or just float.
            # k goes up to 24. 24! is ~ 6e23. Float64 handles this fine.
            fact = 1.0
            for m in range(2, k + 1):
                fact *= m
            norm_sq += q_k * fact
        
        if norm_sq <= 0:
            return 1e100  # Invalid
            
        return -np.log(norm_sq)
        
    # Gradient calculation (numerical is safer for start, but analytical is better)
    # Let's try L-BFGS-B with numerical gradient first for robustness, 
    # or provide an analytical gradient if possible.
    # Given the small dimension (12), numerical gradient is fast enough.
    
    def gradient(z):
        # Central difference
        eps = 1e-6
        grad = np.zeros_like(z)
        f0 = objective(z)
        for i in range(len(z)):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += eps
            z_minus[i] -= eps
            grad[i] = (objective(z_plus) - objective(z_minus)) / (2 * eps)
        return grad

    # Optimization loop
    best_z = z_init.copy()
    best_score = objective(z_init)
    
    # Use L-BFGS-B
    # Bounds for z: x > 0 => z > -inf. 
    # Practical bounds: x should be reasonably sized. 
    # Seed x in [4, 48]. z in [1.4, 3.9].
    # Let's allow a wide range.
    bounds = [(-10, 20)] * 12
    
    try:
        res = minimize(
            objective, 
            z_init, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.success and res.fun < best_score:
            best_z = res.x
            best_score = res.fun
            
    except Exception as e:
        pass
        
    # If time permits, do a local polish with tighter tolerance
    if time.time() < deadline - 60:
        try:
            res2 = minimize(
                objective, 
                best_z, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            if res2.success and res2.fun < best_score:
                best_z = res2.x
                best_score = res2.fun
        except Exception:
            pass

    x_opt = np.exp(best_z)
    
    # Verify positivity and return list
    if np.any(x_opt <= 0):
        return [4.0 * (i + 1) for i in range(12)]
        
    return list(x_opt)
