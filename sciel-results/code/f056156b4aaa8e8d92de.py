import numpy as np
from scipy.optimize import minimize

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = conv.max()
    return max_conv / (total * total)

def _objective(h):
    return _score(h)

# EVOLVE-BLOCK-START
def construct():
    # Start from the constant function (theoretical lower bound 1/N)
    # and attempt to improve via gradient-free optimization.
    # The metric is max_k (sum_i h_i * h_{i+k}) / (sum h_i)^2
    # For nonnegative h, the max is at k=0: sum h_i^2 / (sum h_i)^2
    # This is minimized when all h_i are equal (Cauchy-Schwarz).
    # So the constant function should be optimal.
    # However, let's verify by trying a few perturbations.
    
    h0 = np.ones(N)
    
    # Try a simple local search: perturb a few coordinates and see if score improves
    best_h = h0.copy()
    best_score = _score(best_h)
    
    # Try sinusoidal perturbations
    for freq in [1, 2, 3, 5]:
        for amp in [0.01, 0.05, 0.1, 0.2]:
            h_test = h0 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h_test = np.maximum(h_test, 0)  # enforce nonnegativity
            if h_test.sum() > 0:
                s = _score(h_test)
                if s < best_score:
                    best_score = s
                    best_h = h_test.copy()
    
    # Try triangular/bell shapes
    center = N // 2
    for width in [N//4, N//3, N//2, 2*N//3]:
        x = np.arange(N) - center
        h_test = np.exp(-x**2 / (2 * width**2))
        if h_test.sum() > 0:
            s = _score(h_test)
            if s < best_score:
                best_score = s
                best_h = h_test.copy()
    
    # Try uniform on a subset
    for frac in [0.5, 0.75, 0.9]:
        m = int(frac * N)
        h_test = np.zeros(N)
        start = (N - m) // 2
        h_test[start:start+m] = 1.0
        s = _score(h_test)
        if s < best_score:
            best_score = s
            best_h = h_test.copy()
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
