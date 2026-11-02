"""Rewritten program: continuous optimization with L1-normalized formulation."""
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _score(heights):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup(h*h))."""
    h = np.asarray(heights, dtype=float)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 == 0.0:
        return 0.0
    conv = fftconvolve(h, h, mode='full')
    sup_conv = float(np.max(conv))
    if sup_conv == 0.0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _norm_score(h):
    """Score with L1=1 normalization: L2^2 / sup(conv) when L1=1."""
    l1 = float(np.sum(h))
    if l1 < 1e-12:
        return 0.0
    hn = h / l1
    l2_sq = float(np.dot(hn, hn))
    conv = fftconvolve(hn, hn, mode='full')
    sup_conv = float(np.max(conv))
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / sup_conv

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_score = 0.0
    best_heights = np.ones(N) / N
    
    rng = np.random.RandomState(2024)
    
    # Generate diverse initializations
    inits = []
    
    # Sparse random
    for _ in range(15):
        n_spikes = rng.randint(3, 18)
        h = np.zeros(N)
        indices = rng.choice(N, size=n_spikes, replace=False)
        h[indices] = rng.uniform(0.1, 1.0, size=n_spikes)
        inits.append(h)
    
    # Sidon-like
    for _ in range(8):
        used_sums = set()
        sidon = []
        for x in range(N):
            valid = True
            for y in sidon:
                if (x + y) in used_sums:
                    valid = False
                    break
            if valid:
                for y in sidon:
                    used_sums.add(x + y)
                used_sums.add(2 * x)
                sidon.append(x)
            if len(sidon) >= rng.randint(5, 12):
                break
        h = np.zeros(N)
        for idx in sidon:
            h[idx] = rng.uniform(0.5, 1.5)
        inits.append(h)
    
    # Smooth bumps
    for _ in range(8):
        n_bumps = rng.randint(1, 5)
        h = np.zeros(N)
        x = np.arange(N)
        for _ in range(n_bumps):
            center = rng.uniform(0, N-1)
            width = rng.uniform(1.5, 10)
            amp = rng.uniform(0.5, 2.0)
            h += amp * np.exp(-0.5 * ((x - center) / width)**2)
        inits.append(h)
    
    # Random with sparsification
    for _ in range(8):
        h = rng.uniform(0, 1.0, size=N)
        density = rng.uniform(0.15, 0.5)
        mask = rng.random(N) < density
        h[~mask] = 0.0
        if np.sum(h) < 1e-8:
            h[rng.randint(N)] = 1.0
        inits.append(h)
    
    # Near-uniform perturbations
    for _ in range(5):
        h = np.ones(N) * 0.5 + rng.normal(0, 0.4, size=N)
        h = np.clip(h, 0, None)
        inits.append(h)
    
    # Optimize each with L-BFGS-B on log-heights
    for h_init in inits:
        if time.time() > deadline - 12:
            break
        
        h_init = np.clip(h_init, 0, None)
        if np.sum(h_init) < 1e-10:
            continue
        
        s = _score(h_init)
        if s > best_score:
            best_score = s
            best_heights = h_init.copy()
        
        log_h = np.log(h_init + 1e-10)
        
        def obj(lh):
            hh = np.exp(lh)
            return -_norm_score(hh)
        
        try:
            res = minimize(
                obj, log_h, method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            h_opt = np.exp(res.x)
            h_opt = np.clip(h_opt, 0, None)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_heights = h_opt.copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 5:
        h = best_heights.copy()
        for iteration in range(80):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                for factor in [0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]:
                    h_new = h.copy()
                    h_new[i] = h[i] * factor if h[i] > 1e-10 else factor * 0.05
                    s_new = _score(h_new)
                    if s_new > best_score + 1e-10:
                        best_score = s_new
                        best_heights = h_new.copy()
                        h = h_new.copy()
                        improved = True
                        break
            if not improved:
                break
    
    return [float(x) for x in best_heights]
# EVOLVE-BLOCK-END
