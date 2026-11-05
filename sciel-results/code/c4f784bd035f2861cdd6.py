"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    timeout = 110  # 10 second safety margin
    
    def objective(h):
        h = np.asarray(h)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        return abs(2 * N * np.max(conv) / (s ** 2))
    
    # Try several initializations and optimize
    best_value = 2.0  # seed value
    best_h = np.ones(N)
    
    # Try different parameterizations
    attempts = [
        # 1. Small perturbations from uniform
        np.ones(N) + 0.1 * np.random.randn(N),
        # 2. Gaussian-shaped
        np.exp(-((np.arange(N) - N/2) ** 2) / (2 * (N/4) ** 2)),
        # 3. Bimodal
        np.exp(-((np.arange(N) - N*0.25) ** 2) / (2 * (N/20) ** 2)) +
        np.exp(-((np.arange(N) - N*0.75) ** 2) / (2 * (N/20) ** 2)),
        # 4. Cosine taper
        np.abs(np.cos(np.pi * np.arange(N) / N)),
    ]
    
    for init in attempts:
        if time.time() - start_time > timeout:
            break
        
        # Normalize to have sum = N for consistent scale
        init = init / np.sum(init) * N
        init = np.clip(init, -1e-6, None)  # Keep non-negative for simplicity
        
        result = minimize(
            objective,
            init,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < best_value:
            best_value = result.fun
            best_h = result.x
    
    return best_h
# EVOLVE-BLOCK-END
