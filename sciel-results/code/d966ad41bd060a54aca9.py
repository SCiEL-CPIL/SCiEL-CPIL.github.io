"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    Approach: Try several structured families and also run a local optimization
    to minimize max(autoconv) subject to sum(h)=1.
    """
    def value_of(h):
        s = np.sum(h)
        if abs(s) < 1e-15:
            return np.inf
        ac = np.convolve(h, h)
        return abs(2 * N * np.max(ac) / (s * s))
    
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    # Try several structured candidates
    candidates = []
    i = np.arange(N)
    
    # 1. Uniform (seed)
    candidates.append(np.ones(N))
    
    # 2. Sine wave (one period) with small offset to ensure nonzero sum
    candidates.append(np.sin(2 * np.pi * i / N) + 0.05)
    
    # 3. Cosine wave
    candidates.append(np.cos(2 * np.pi * i / N) + 0.05)
    
    # 4. Linear ramp
    candidates.append(i.astype(float) - (N - 1) / 2)
    
    # 5. Quadratic
    candidates.append((i.astype(float) - (N - 1) / 2) ** 2 - N**2/12)
    
    # 6. Gaussian-like bumps
    center = (N - 1) / 2
    for sigma in [N/4, N/6, N/8]:
        candidates.append(np.exp(-((i - center) ** 2) / (2 * sigma ** 2)))
    
    # 7. Bump: positive in middle, negative on edges
    mask_mid = (i >= N//4) & (i < 3*N//4)
    h_bump = np.zeros(N)
    h_bump[mask_mid] = 1
    h_bump[~mask_mid] = -0.3
    candidates.append(h_bump)
    
    # 8. Two-lobe: positive on left, negative on right with offset
    h_two = np.zeros(N)
    h_two[:N//2] = 1
    h_two[N//2:] = -0.8
    candidates.append(h_two)
    
    # 9. Three-lobe
    h_three = np.zeros(N)
    h_three[:N//3] = -0.5
    h_three[N//3:2*N//3] = 1
    h_three[2*N//3:] = -0.5
    candidates.append(h_three)
    
    # 10. Smooth sign-changing: sin with multiple periods
    for freq in [2, 3, 4]:
        candidates.append(np.sin(2 * np.pi * freq * i / N) + 0.02)
    
    # Evaluate all candidates
    for h in candidates:
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        h_norm = h / s  # normalize so sum = 1
        v = value_of(h_norm)
        if v < best_val:
            best_val = v
            best_h = h_norm.copy()
    
    # Try local optimization from the best candidate
    def objective(h):
        ac = np.convolve(h, h)
        return np.max(ac)
    
    def constraint_sum(h):
        return np.sum(h) - 1.0
    
    try:
        h0 = best_h.copy()
        result = minimize(objective, h0, method='SLSQP', 
                          constraints=[{'type': 'eq', 'fun': constraint_sum}],
                          options={'maxiter': 300, 'ftol': 1e-12})
        if hasattr(result, 'x') and result.x is not None:
            h_opt = result.x
            s = np.sum(h_opt)
            if abs(s) > 1e-10:
                h_opt = h_opt / s
                v = value_of(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
    except Exception:
        pass
    
    # Final check
    s = np.sum(best_h)
    if abs(s) < 1e-10:
        return np.ones(N)
    
    return best_h
