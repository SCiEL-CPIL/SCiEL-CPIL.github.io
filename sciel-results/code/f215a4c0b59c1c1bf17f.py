"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    Key insight: The averaging argument gives value >= 2N/(2N-1) ≈ 1.0025,
    so 2.0 is NOT a tight lower bound. We can do better by flattening the autoconvolution.
    
    Approach: Multi-start L-BFGS-B with soft-max objective, fixing sum(h)=1.
    """
    def value_of(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        ac = np.convolve(h, h)
        return abs(2.0 * N * np.max(ac) / (s * s))
    
    best_h = np.ones(N)
    best_val = value_of(best_h)
    
    # Structured initializations
    x = np.arange(N)
    initializations = [np.ones(N) / N]
    
    # Sine/cosine perturbations
    for freq in [1, 2, 4]:
        for amp in [0.3, 0.5, 0.7]:
            h = 1.0 + amp * np.sin(2 * np.pi * freq * x / N)
            if abs(np.sum(h)) > 1e-10:
                initializations.append(h / np.sum(h))
            h = 1.0 + amp * np.cos(2 * np.pi * freq * x / N)
            if abs(np.sum(h)) > 1e-10:
                initializations.append(h / np.sum(h))
    
    # Gaussian windows
    center = N / 2.0
    for sigma in [N/4, N/3, N/2, 2*N/3]:
        g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        initializations.append(g / np.sum(g))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(8):
        h0 = rng.randn(N)
        if abs(np.sum(h0)) > 1e-10:
            initializations.append(h0 / np.sum(h0))
    
    for h0 in initializations:
        h0 = h0 / np.sum(h0)
        
        def objective(h):
            ac = np.convolve(h, h)
            t = 300.0
            m = np.max(ac)
            return m + np.log(np.sum(np.exp(t * (ac - m)))) / t
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
        try:
            result = minimize(objective, h0, method='L-BFGS-B', constraints=constraints,
                              options={'maxiter': 300, 'ftol': 1e-10})
            h_opt = result.x / np.sum(result.x)
            v = value_of(h_opt)
            if v < best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            continue
    
    return best_h
