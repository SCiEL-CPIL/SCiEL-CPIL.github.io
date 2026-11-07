import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _aux_obj(h, lam):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    return l2 - lam * norm_1 * norm_inf

def _dinkelbach_optimize(h0, deadline, max_iter=20):
    """
    Dinkelbach's algorithm using Nelder-Mead for the auxiliary problem.
    """
    h = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = h.sum()
    if s < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
        
    lam = _compute_value(h)
    
    # Nelder-Mead options
    # We want to maximize aux, so we minimize -aux
    def neg_aux(h_raw):
        hh = np.maximum(h_raw, 0.0)
        s = hh.sum()
        if s < 1e-12:
            return 1e10
        hh = hh / s
        return -_aux_obj(hh, lam)
        
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Optimize auxiliary
        try:
            res = minimize(neg_aux, h.copy(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            h_new = np.maximum(res.x, 0.0)
            s = h_new.sum()
            if s > 1e-12:
                h_new = h_new / s
                new_val = _compute_value(h_new)
                if new_val > lam * (1 + 1e-8):
                    h = h_new
                    lam = new_val
                else:
                    # Converged or no improvement
                    break
            else:
                break
        except Exception:
            break
            
    return h, _compute_value(h)

def _gen_starts(rng):
    """Generate diverse structured starts."""
    starts = []
    idx = np.arange(N)
    x = idx / (N - 1) - 0.5
    
    # Uniform
    starts.append(np.ones(N))
    
    # Gaussians
    for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
        starts.append(np.exp(-x**2 / (2 * s**2)))
        
    # Super/sub-Gaussian
    for p in [0.5, 1, 1.5, 2, 3, 4]:
        for s in [0.1, 0.2, 0.4, 0.7]:
            starts.append(np.exp(-np.abs(x) ** p / s))
            
    # Compact support
    for w in [0.1, 0.2, 0.3, 0.5]:
        starts.append(np.maximum(0, 1 - np.abs(x) / w))
        
    # Plateau
    for w in [0.1, 0.2, 0.3]:
        starts.append(np.where(np.abs(x) < w, 1.0, 0.0) + 1e-6)
        
    # Bimodal
    for c in [0.15, 0.25, 0.35]:
        for s in [0.05, 0.1, 0.15]:
            starts.append(np.exp(-(x - c)**2 / (2*s**2)) + np.exp(-(x + c)**2 / (2*s**2)))
            
    # Tri-modal
    for s in [0.08, 0.12, 0.2]:
        starts.append(np.exp(-(x - 0.4)**2/(2*s**2)) + np.exp(-x**2/(2*s**2)) + np.exp(-(x + 0.4)**2/(2*s**2)))
        
    # Lorentzian
    for a in [2, 5, 10, 20]:
        starts.append(1.0 / (1 + (a * x)**2))
        
    # Power-law
    for a in [0.5, 1, 2, 3]:
        starts.append(1.0 / (1 + np.abs(x) / 0.2) ** a)
        
    # Random smooth
    for _ in range(20):
        nfreq = int(rng.integers(2, 8))
        coeffs = rng.standard_normal(nfreq)
        phases = rng.uniform(0, 2*np.pi, nfreq)
        h_f = np.ones(N)
        for k in range(1, nfreq + 1):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*x + phases[k-1])
        starts.append(np.abs(h_f) + 0.02)
        
    # Random
    for _ in range(20):
        starts.append(np.abs(rng.standard_normal(N)) + 0.02)
        
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    rng = np.random.default_rng(50)
    starts = _gen_starts(rng)
    
    best_h = np.ones(N) / N
    best_val = _compute_value(best_h)
    
    def try_candidate(h0):
        nonlocal best_h, best_val
        if time.time() > deadline - 10:
            return
        h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
        s = h0.sum()
        if s < 1e-12:
            return
        h0 = h0 / s
        
        # Dinkelbach with Nelder-Mead
        try:
            h_d, v_d = _dinkelbach_optimize(h0, deadline)
            if v_d > best_val:
                best_val = v_d
                best_h = h_d.copy()
        except Exception:
            pass
        
        # Optional: A quick polish with a different method if time permits
        # For now, we rely on the robustness of Nelder-Mead within Dinkelbach.
        # If we have time, we could try L-BFGS-B on the ratio directly, but it's risky.
        
    # Phase 1: Try all structured starts
    for h0 in starts:
        if time.time() > deadline - 20:
            break
        try_candidate(h0)
        
    # Phase 2: If time remains, try more random restarts
    if time.time() < deadline - 30:
        for seed in [1, 2, 3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if time.time() > deadline - 20:
                break
            rng2 = np.random.default_rng(seed)
            for _ in range(5):
                if time.time() > deadline - 20:
                    break
                h0 = np.abs(rng2.standard_normal(N)) + 0.02
                try_candidate(h0)
                
    # Phase 3: Final polish
    if time.time() < deadline - 5:
        try:
            # Run Dinkelbach again on the best found to ensure convergence
            h_f, v_f = _dinkelbach_optimize(best_h, deadline, max_iter=5)
            if v_f > best_val:
                best_val = v_f
                best_h = h_f.copy()
        except Exception:
            pass

    # Ensure validity
    best_h = np.maximum(np.asarray(best_h, dtype=float), 0.0)
    s = best_h.sum()
    if s < 1e-12:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
