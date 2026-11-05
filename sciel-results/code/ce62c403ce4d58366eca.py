import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    def generate_h(params):
        phi0, mu = params
        k = np.arange(N)
        # Center the chirp to potentially reduce boundary effects?
        # k_centered = k - N/2
        # h = np.cos(phi0 + mu * k_centered**2)
        # Original:
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

    # 1. Global Search using Differential Evolution
    # Search over phi0 and mu.
    # Use a linear scale for mu but constrained to positive values to avoid symmetry issues?
    # Or allow negative mu? cos is even in mu if phi0 adjusted?
    # cos(phi0 - mu k^2) is same as cos(phi0 + mu k^2) with phase shift?
    # No, cos(A-B) = cos(A+B) only if B=0 or specific conditions.
    # But cos is even: cos(-x) = cos(x).
    # So mu > 0 is sufficient if we can adjust phi0.
    
    bounds = [(0, 2 * np.pi), (1e-5, 10.0)]
    
    result_de = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=200, 
        tol=1e-6, 
        popsize=20, 
        init='latinhypercube',
        polish=False 
    )
    
    best_val = result_de.fun
    best_params = result_de.x
    
    # 2. Local Refinement
    result_nm = minimize(
        objective, 
        best_params, 
        method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000}
    )
    
    if result_nm.fun < best_val:
        best_val = result_nm.fun
        best_params = result_nm.x

    # 3. Random restarts
    rng = np.random.default_rng(42)
    for _ in range(20):
        p0 = [rng.uniform(0, 2*np.pi), rng.uniform(1e-5, 10.0)]
        res = minimize(
            objective, 
            p0, 
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 2000}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # Final check
    h = generate_h(best_params)
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-6:
        return np.ones(N)
        
    conv_h = np.convolve(h, h)
    max_conv = np.max(conv_h)
    val = abs(2 * N * max_conv / (sum_h ** 2))
    
    if val < 2.0:
        return h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
