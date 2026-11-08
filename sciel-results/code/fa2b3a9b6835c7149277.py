import numpy as np
from scipy.optimize import differential_evolution, minimize


def construct():
    # Hermite polynomial values at 0 for even degrees (physicists' convention)
    # H_0(0) = 1, H_4(0) = -3, H_8(0) = 105, H_12(0) = -10395
    h0, h4, h8, h12 = 1.0, -3.0, 105.0, -10395.0
    
    # n + 1/2 for degrees 0, 4, 8, 12
    moments = np.array([0.5, 4.5, 8.5, 12.5])
    
    def objective(coeffs):
        """coeffs = [c0, c4, c8], c12 derived from g(0)=0"""
        c0, c4, c8 = coeffs
        # g(0) = c0*h0 + c4*h4 + c8*h8 + c12*h12 = 0
        # c12 = -(c0*h0 + c4*h4 + c8*h8) / h12
        c12 = -(c0 * h0 + c4 * h4 + c8 * h8) / h12
        c = np.array([c0, c4, c8, c12])
        
        c2 = c**2
        q = np.sum(c2)
        s = np.dot(c2, moments)
        
        if q < 1e-14:
            return 1e10
        return (s / q)**2

    # Define bounds for the optimizer
    # The scale of coefficients doesn't matter, only the ratios.
    # We can bound them to [-10, 10] or similar.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use differential evolution for global optimization
    result = differential_evolution(objective, bounds, maxiter=1000, tol=1e-12, seed=42)
    
    # Also try a local refinement from the DE result and the seed
    candidates = [result.x, np.array([-1.13, 0.04, 0.0003]), np.array([1.0, 0.0, 0.0])]
    
    best_x = None
    best_f = np.inf
    
    for x0 in candidates:
        # Refine with Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            
    return np.array(best_x)
