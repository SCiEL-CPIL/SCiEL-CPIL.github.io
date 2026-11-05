import numpy as np
from scipy.optimize import minimize

N = 400

def objective(params):
    h = np.exp(params)  # Ensure positivity via log-parameterization
    s = np.sum(h)
    if s == 0:
        return 1e10
    c = np.convolve(h, h)
    max_c = np.max(c)
    return abs(2 * N * max_c / (s ** 2))

def construct():
    # Start from log(ones) = 0
    x0 = np.zeros(N)
    
    # Use L-BFGS-B for optimization with bounds to prevent explosion
    # We don't really need bounds if we use exp, but let's keep parameters in a reasonable range
    # Actually, since h = exp(x), and we only care about the ratio, the scale of x doesn't matter for the objective value itself, 
    # but it affects the gradient magnitude. 
    # However, the objective is scale-invariant: f(c*h) = f(h).
    # This creates a flat direction in the parameter space. 
    # To handle this, we can fix one parameter or use a constraint.
    # Alternatively, just let the optimizer handle it. The gradient will be small in the scale direction.
    # A better approach: normalize h inside the objective? No, that changes the gradient.
    # Let's just minimize the raw objective.
    
    # To improve convergence, we can use a small amount of regularization to keep parameters near 0?
    # Or just rely on L-BFGS-B.
    
    # Let's try a simple approach first.
    # Since the problem is symmetric, we might get stuck in symmetric local minima.
    # Random restarts can help.
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # Perform a few random restarts with L-BFGS-B
    for restart in range(5):
        if restart == 0:
            x_start = np.zeros(N)
        else:
            # Random small perturbation in log-space
            x_start = np.random.randn(N) * 0.1
        
        # We need to be careful with the scale invariance.
        # Let's fix the sum of h to be N to remove the scale invariance.
        # If we fix sum(h) = N, then we are optimizing over the shape.
        # Constraint: sum(exp(x)) = N.
        # This is a non-linear constraint.
        
        # Alternative: Just optimize the original objective.
        # The optimizer might drift in the scale direction, but the value won't change.
        # This is fine.
        
        res = minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_h = np.exp(res.x)
            
    # Normalize the final result to ensure sum is reasonable (not required by task, but good practice)
    # The task says "Choose real... heights". It doesn't restrict the sum.
    # The value depends on the ratio, so scaling doesn't matter.
    # However, numerical stability might be better if h is not too large or small.
    # Let's normalize best_h to have sum N.
    if np.sum(best_h) > 0:
        best_h = best_h * (N / np.sum(best_h))
        
    return best_h
