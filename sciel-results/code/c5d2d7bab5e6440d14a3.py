import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # We want to minimize max(convolve(h, h)) / sum(h)^2
    # By fixing sum(h) = 1, we minimize max(convolve(h, h))
    # A good proxy for minimizing the max is minimizing the sum of squares of the convolution
    # which flattens the distribution of coefficients.
    
    def objective(h):
        c = np.convolve(h, h)
        # Minimize the sum of squares to flatten the peak
        val = np.sum(c**2)
        # Penalty to keep sum(h) close to 1
        penalty = 100.0 * (np.sum(h) - 1.0)**2
        return val + penalty

    # Start with a constant vector normalized to sum 1
    x0 = np.ones(N) / N
    
    # Optimize
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12})
    
    h_opt = res.x
    
    # Verify sum is close to 1
    # If sum is not 1, the value calculation in the evaluator uses the actual sum.
    # But we minimized assuming sum=1.
    # The ratio is scale invariant, so we can return h_opt as is.
    
    return h_opt
