import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss

def construct():
    t_start = time.time()
    deadline = 205.0  # 5s margin from 210s
    
    # Degrees
    degrees = [0, 4, 8, 12]
    
    # Precompute Hermite values on a finer grid for the sup-norm
    # Use a wider range and more points for better accuracy
    x_grid = np.linspace(-8, 8, 4000)
    H_grid = np.array([eval_hermite(d, x_grid) for d in degrees])
    
    # Gauss-Hermite for norm with more points
    n_points = 300
    y_points, y_weights = hermgauss(n_points)
    x_points = y_points / np.sqrt(2 * np.pi)
    H_points = np.array([eval_hermite(d, x_points) for d in degrees])
    
    def get_c12(c0, c4, c8):
        # c0*1 + c4*(-3) + c8*(105) + c12*(10395) = 0
        return (-c0 + 3*c4 - 105*c8) / 10395.0

    def objective(params):
        c0, c4, c8 = params
        c12 = get_c12(c0, c4, c8)
        coeffs = np.array([c0, c4, c8, c12])
        
        # Calculate P on grid using matrix multiplication for speed
        P_grid = H_grid.T @ coeffs
        
        # Calculate Sup of |P| * exp(-pi x^2)
        gauss_factor = np.exp(-np.pi * x_grid**2)
        sup_val = np.max(np.abs(P_grid * gauss_factor))
        
        # Calculate Norm using matrix multiplication
        P_points = H_points.T @ coeffs
        norm_sq = np.sum((P_points**2) * y_weights) / np.sqrt(2 * np.pi)
        
        if norm_sq < 1e-12:
            return 1e12
            
        return (sup_val**2) / norm_sq

    # Seed coefficients from the initial program
    seed_params = np.array([-113/100, 1/25, 1/3240])
    
    # Evaluate seed first
    best_score = objective(seed_params)
    best_params = seed_params.copy()
    
    # Try multiple optimization methods and starting points
    starting_points = [
        seed_params,
        seed_params + np.array([0.1, 0.01, 0.001]),
        seed_params + np.array([-0.1, -0.01, -0.001]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-1.1, 0.03, 0.0002]),
    ]
    
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        for x0 in starting_points:
            if time.time() - t_start > deadline - 5:
                break
            try:
                if method == 'Nelder-Mead':
                    res = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
                else:
                    res = minimize(objective, x0, method='Powell',
                                  options={'maxiter': 3000, 'xtol': 1e-9, 'ftol': 1e-11})
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        if time.time() - t_start > deadline - 5:
            break
    
    # Return the best parameters found
    return np.array(best_params)
