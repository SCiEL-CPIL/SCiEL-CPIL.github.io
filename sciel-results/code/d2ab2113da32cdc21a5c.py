import numpy as np
from scipy.optimize import minimize
import time

# Precompute factorials up to 48! for the integrals
# I1 = sum c_k * k!
# I2 = sum_i sum_j c_i c_j * (i+j)!
MAX_DEG = 48
factorials = np.array([1.0] * (MAX_DEG + 1))
for k in range(1, MAX_DEG + 1):
    factorials[k] = factorials[k - 1] * k

def get_coefficients(roots):
    """
    Computes coefficients of P(x) = prod (x - r_i)^2
    roots: array of 12 values
    returns: array of 25 coefficients c_0 ... c_24 for x^0 ... x^24
    """
    # Start with 1
    coeffs = np.array([1.0])
    
    for r in roots:
        # (x - r)^2 = x^2 - 2r x + r^2
        # Multiplying current poly by this quadratic
        # new_coeffs[k] = old_coeffs[k-2]*1 + old_coeffs[k-1]*(-2r) + old_coeffs[k]*(r^2)
        # Note: indexing. coeffs[i] is coeff of x^i.
        
        new_coeffs = np.zeros(len(coeffs) + 2)
        
        # Term x^2 * old
        new_coeffs[2:] += coeffs
        # Term -2r x * old
        new_coeffs[1:-1] += -2.0 * r * coeffs
        # Term r^2 * old
        new_coeffs[0:-2] += r**2 * coeffs
        
        coeffs = new_coeffs
        
    return coeffs

def objective(r):
    """
    Computes the bound value to be minimized.
    r: array of 12 roots
    """
    # Ensure roots are real and positive (optimizer might go negative, but we use bounds)
    # Sort roots to ensure consistent polynomial construction (though symmetric)
    r_sorted = np.sort(r)
    
    coeffs = get_coefficients(r_sorted)
    
    # I1 = sum c_k * k!
    # coeffs has length 25 (0 to 24)
    # factorials has length 49 (0 to 48)
    I1 = np.dot(coeffs, factorials[:25])
    
    # I2 = sum_i sum_j c_i c_j (i+j)!
    # This is equivalent to coeffs.T @ M @ coeffs where M_ij = (i+j)!
    # We can compute this efficiently.
    # Let's just do the double sum since 25x25 is small.
    I2 = 0.0
    for i in range(25):
        if coeffs[i] == 0:
            continue
        for j in range(25):
            if coeffs[j] == 0:
                continue
            I2 += coeffs[i] * coeffs[j] * factorials[i + j]
            
    if I1 == 0:
        return 1e100
        
    return I2 / (I1 * I1)

def construct():
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    # Initial guess: Roots of Laguerre L_12^0(x)
    # Or simply use the seed values
    # Seed: [4.0 * (i + 1) for i in range(12)]
    r0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # We can also try roots of L_12^alpha for some alpha as a better start
    # But seed is fine.
    
    # Bounds: roots must be positive
    bounds = [(1e-6, None)] * 12
    
    best_r = r0
    best_val = objective(r0)
    
    # Optimization
    try:
        # Use L-BFGS-B for bounded optimization
        res = minimize(
            objective, 
            r0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_val:
            best_r = res.x
            best_val = res.fun
        else:
            # If it failed, check if we are still within time
            pass
            
    except Exception as e:
        pass
    
    # Check time
    if time.time() < deadline - 10:
        # Try a local refinement or different start if needed?
        # For now, one run of L-BFGS-B should be sufficient and fast.
        pass
        
    # Return the list of roots
    return list(best_r)
