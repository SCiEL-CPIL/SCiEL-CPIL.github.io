import numpy as np
from scipy.optimize import differential_evolution
from numpy.polynomial.laguerre import laggauss

def construct():
    # Objective: Minimize the L4 norm squared of the polynomial P(x) = prod (x-xi)^2
    # under the Laguerre weight e^-x.
    # Integral I = int_0^inf e^-x (prod (x-xi)^2)^2 dx = int e^-x prod (x-xi)^4 dx
    
    # Use Gaussian Quadrature for the integral.
    # Degree of integrand is 4*12 = 48.
    # Laguerre quadrature of order N is exact for degree 2N-1.
    # 2N-1 >= 48 => N >= 24.5 => N=25.
    # Use N=50 for safety.
    n_quad = 50
    x_nodes, w_nodes = laggauss(n_quad)
    
    def cost_func(roots):
        # roots: (12,)
        # Calculate prod (x_nodes - xi)^4
        # To avoid overflow/underflow in the product, we can use logs if values are extreme,
        # but for x in [0, 100] and roots in [0, 100], values are manageable.
        # However, (100-0)^4 = 1e8. Product of 12 terms = 1e96. Sum of 50 terms = 1e98.
        # This fits in float64 (max ~1e308).
        
        diff = x_nodes[:, None] - roots[None, :]
        # Clip small negative values from numerical noise if any, though x_nodes > 0 and roots > 0.
        # Actually, roots can be very small.
        
        prod_diff = np.prod(diff**4, axis=1)
        integral_val = np.sum(w_nodes * prod_diff)
        return integral_val

    # Bounds for roots: Positive.
    # Typical Laguerre roots for n=12 are between ~0.1 and ~40.
    # Let's set bounds [1e-4, 100].
    bounds = [(1e-4, 100.0) for _ in range(12)]
    
    # Use Differential Evolution for global optimization.
    # popsize=15, maxiter=200 should be fast enough.
    # tol=1e-6
    
    result = differential_evolution(cost_func, bounds, seed=42, maxiter=100, popsize=15, tol=1e-6, polish=True)
    
    if result.success:
        return list(result.x)
    else:
        # Fallback: If DE fails, try L-BFGS-B from Gaussian nodes as in original?
        # Or just return Gaussian nodes.
        x_init, _ = laggauss(12)
        return list(x_init)
