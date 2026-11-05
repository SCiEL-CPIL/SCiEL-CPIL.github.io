import numpy as np
import time

def construct():
    N = 400
    start = time.time()
    deadline = start + 140.0
    
    def eval_h(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    best_h = np.ones(N)
    best_val = eval_h(best_h)
    
    # Build diverse single-mode patterns
    patterns = []
    
    # Square waves with various periods
    for p in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 400]:
        half = max(1, p // 2)
        sq = np.zeros(N)
        for i in range(N):
            if (i % p) < half:
                sq[i] = 1.0
            else:
                sq[i] = -1.0
        patterns.append(sq)
    
    # Cosine modes
    for k in range(1, 201):
        patterns.append(np.cos(2.0 * np.pi * k * np.arange(N) / N))
    
    # Sine modes
    for k in range(1, 201, 5):
        patterns.append(np.sin(2.0 * np.pi * k * np.arange(N) / N))
    
    # For each pattern: h = 1/N + a * (pat - mean(pat)), so sum(h)=1
    # Then value = 2*N*|max(conv(h,h))|
    for pat in patterns:
        if time.time() > deadline:
            break
        mean_pat = np.mean(pat)
        centered = pat - mean_pat
        max_centered = np.max(np.abs(centered))
        if max_centered < 1e-10:
            continue
        
        a_range = np.linspace(-3.0, 3.0, 61)
        
        for a in a_range:
            if time.time() > deadline:
                break
            h = 1.0 / N + a * centered
            conv = np.convolve(h, h)
            val = abs(2.0 * N * np.max(conv))
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Local refinement with random perturbations
    if best_val < 2.0 and time.time() < deadline - 5:
        h = best_h.copy()
        rng = np.random.default_rng(123)
        
        for trial in range(500):
            if time.time() > deadline:
                break
            h_trial = h.copy()
            n_perturb = int(rng.integers(3, 10))
            idx = rng.choice(N, size=n_perturb, replace=False)
            for i in idx:
                h_trial[i] += rng.uniform(-0.2, 0.2)
            s = np.sum(h_trial)
            if abs(s) < 1e-10:
                continue
            conv = np.convolve(h_trial, h_trial)
            val = abs(2.0 * N * np.max(conv) / (s * s))
            if val < best_val:
                best_val = val
                h = h_trial.copy()
                best_h = h_trial.copy()
    
    return best_h
