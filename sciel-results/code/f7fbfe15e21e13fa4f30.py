import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    def generate_h(params):
        phi0, mu = params
        k = np.arange(N)
        h = np.cos(phi0 + mu * k**2)
        return h

    def objective(params):
        h = generate_h(params)
        sum_h = np.sum(h)
        if abs(sum_h) < 1e-8:
            return 1e10
        conv_h = np.convolve(h, h)
        max_conv = np.max(conv_h)
        value = abs(2 * N * max_conv / (sum_h ** 2))
        return value

    # Define bounds
    # mu can be very small or large. Log space is better for DE if we transform,
    # but DE works on linear bounds. We use a wide linear bound.
    bounds = [(0, 2 * np.pi), (1e-6, 100.0)]
    
    best_val = float('inf')
    best_params = None
    
    # Strategy: Multiple DE runs with different seeds and population sizes
    # to ensure we explore the landscape well.
    # We also do a final NM refinement.
    
    # Run 1: Standard DE
    # Using a larger population and more iterations to find the global min
    # 120s limit is generous for N=400.
    de_res1 = differential_evolution(
        objective, bounds, seed=12345, maxiter=300, tol=1e-8,
        popsize=30, init='latinhypercube'
    )
    
    if de_res1.fun < best_val:
        best_val = de_res1.fun
        best_params = de_res1.x
        
    # Run 2: DE with slightly different seed to check for local minima
    de_res2 = differential_evolution(
        objective, bounds, seed=99999, maxiter=300, tol=1e-8,
        popsize=30, init='latinhypercube'
    )
    
    if de_res2.fun < best_val:
        best_val = de_res2.fun
        best_params = de_res2.x
        
    # Run 3: Local refinement around the best found using DE with tight bounds
    # This helps if the global DE stopped early or got stuck in a valley.
    if best_params is not None:
        phi0_best, mu_best = best_params
        # Tight bounds around the best solution
        local_bounds = [
            (max(0, phi0_best - 0.5), min(2*np.pi, phi0_best + 0.5)),
            (max(1e-6, mu_best * 0.5), min(100.0, mu_best * 2.0))
        ]
        # Ensure bounds are valid (lower < upper)
        if local_bounds[0][0] < local_bounds[0][1] and local_bounds[1][0] < local_bounds[1][1]:
            de_res3 = differential_evolution(
                objective, local_bounds, seed=777, maxiter=200, tol=1e-10,
                popsize=20, init='latinhypercube'
            )
            if de_res3.fun < best_val:
                best_val = de_res3.fun
                best_params = de_res3.x
                
    # Final Nelder-Mead Refinement
    # NM is good for final polishing
    if best_params is not None:
        result_nm = minimize(
            objective, best_params, method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000}
        )
        if result_nm.fun < best_val:
            best_val = result_nm.fun
            best_params = result_nm.x

    # Final check and construction
    if best_params is None:
        return np.ones(N)
        
    h = generate_h(best_params)
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-6:
        return np.ones(N)
        
    conv_h = np.convolve(h, h)
    max_conv = np.max(conv_h)
    val = abs(2 * N * max_conv / (sum_h ** 2))
    
    # Compare with seed (value 2.0)
    if val < 2.0:
        return h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
