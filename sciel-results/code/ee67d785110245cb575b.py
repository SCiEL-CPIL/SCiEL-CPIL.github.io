import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import eval_hermite
import time

def _get_hermite_values(x, coeffs):
    """
    Compute g(x) = sum c_k H_k(x) exp(-x^2/2)
    coeffs: [a0, a4, a8, a12]
    """
    a0, a4, a8, a12 = coeffs
    # H0, H4, H8, H12
    # Using eval_hermite
    h0 = eval_hermite(0, x)
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    
    poly_part = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    gauss_part = np.exp(-0.5 * x**2)
    return poly_part * gauss_part

def _compute_norm2_sq(coeffs):
    """
    Compute ||g||_2^2 = sum c_k^2 * sqrt(pi) * k!
    """
    a0, a4, a8, a12 = coeffs
    c = np.array([a0, a4, a8, a12])
    k = np.array([0, 4, 8, 12])
    # sqrt(pi) * k!
    # 0! = 1, 4! = 24, 8! = 40320, 12! = 479001600
    facts = np.array([1.0, 24.0, 40320.0, 479001600.0])
    return np.sqrt(np.pi) * np.sum(c**2 * facts)

def _objective(coeffs):
    """
    Minimize (||g||_1^2) / (||g||_2^2)
    """
    a0, a4, a8 = coeffs
    # Derive a12
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    c = [a0, a4, a8, a12]
    
    n2_sq = _compute_norm2_sq(c)
    if n2_sq < 1e-10:
        return 1e10
        
    # Compute L1 norm
    # g(x) is even?
    # H0 is even, H4 is even, H8 is even, H12 is even.
    # So g(x) is even.
    # Integral from -inf to inf is 2 * Integral from 0 to inf.
    def integrand(x):
        val = _get_hermite_values(x, c)
        return abs(val)
        
    try:
        l1_half, err = quad(integrand, 0, np.inf, limit=100)
        l1 = 2 * l1_half
    except Exception:
        return 1e10
        
    return (l1 * l1) / n2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # 10s margin
    
    # Seed
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _objective(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Multi-start
    np.random.seed(42)
    
    # Define a range for search
    # a0 ~ -1, a4 ~ 0.04, a8 ~ 0.0003
    # Let's try a wider range initially
    n_starts = 50
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
            
        # Random start
        # Log-uniform or uniform?
        # Let's try uniform in log-space for magnitude, random sign
        # Or just uniform in a box around the seed
        if i == 0:
            x0 = seed_coeffs
        else:
            # Perturb seed
            x0 = seed_coeffs + np.random.normal(0, 0.5, 3)
            # Scale perturbations for higher degrees
            x0[1] += np.random.normal(0, 0.01, 1)[0]
            x0[2] += np.random.normal(0, 0.0001, 1)[0]
            
            # Or random restarts from scratch
            if i % 5 == 0:
                x0 = np.array([
                    np.random.uniform(-2, 2),
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(-0.01, 0.01)
                ])
        
        try:
            # Use Nelder-Mead for robustness
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
                
            # Polish with L-BFGS-B using finite differences
            if time.time() > deadline - 5:
                break
            res2 = minimize(_objective, best_coeffs, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12})
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
                
        except Exception as e:
            pass

    return best_coeffs
