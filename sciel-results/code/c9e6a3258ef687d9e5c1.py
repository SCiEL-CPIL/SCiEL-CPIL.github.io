import time
import numpy as np

N = 600

def compute_metric(h):
    """Compute normalized maximum autoconvolution."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    hn = h / s
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return conv.max()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    # Initialize with uniform
    h_best = np.ones(N)
    best_val = compute_metric(h_best)
    
    # Phase 1: Try diverse structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    center = N // 2
    tri = np.array([max(0, 1.0 - abs(i - center) / center) for i in range(N)])
    candidates.append(tri)
    
    # Gaussian
    x = np.linspace(-3, 3, N)
    gauss = np.exp(-x**2)
    candidates.append(gauss)
    
    # Raised cosine
    rc = 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N)))
    candidates.append(rc)
    
    # Inverse Gaussian
    inv_gauss = 1.0 - 0.3 * np.exp(-3.0 * np.linspace(-1, 1, N)**2)
    inv_gauss = np.maximum(inv_gauss, 0)
    candidates.append(inv_gauss)
    
    # Two-peak
    two = np.zeros(N)
    two[N//4] = 1.0
    two[3*N//4] = 1.0
    candidates.append(two)
    
    # Plateaus
    for w_frac in [0.3, 0.5, 0.7, 0.9]:
        w = int(N * w_frac)
        plat = np.zeros(N)
        plat[N//2 - w//2 : N//2 + w//2 + 1] = 1.0
        candidates.append(plat)
    
    # Ramp
    ramp = np.linspace(1, 0, N)
    ramp[ramp < 0] = 0
    candidates.append(ramp)
    
    # Evaluate candidates
    for c in candidates:
        if elapsed() > deadline - 30:
            break
        c = np.maximum(c, 0)
        if c.sum() < 1e-15:
            continue
        val = compute_metric(c)
        if val < best_val:
            best_val = val
            h_best = c.copy()
    
    # Phase 2: Simulated annealing with coordinate updates
    if elapsed() < deadline - 20:
        T = 0.01
        cooling = 0.999
        h_current = h_best.copy()
        current_val = best_val
        
        # Precompute for speed
        for iteration in range(50000):
            if elapsed() > deadline - 10:
                break
            
            # Pick a random coordinate to perturb
            i = np.random.randint(N)
            old_val = h_current[i]
            
            # Try a multiplicative perturbation
            factor = np.random.choice([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0])
            new_val = max(0.0, old_val * factor)
            
            if new_val == old_val:
                continue
            
            h_current[i] = new_val
            new_metric = compute_metric(h_current)
            
            delta = new_metric - current_val
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_val = new_metric
                if current_val < best_val:
                    best_val = current_val
                    h_best = h_current.copy()
            else:
                h_current[i] = old_val
            
            T *= cooling
    
    # Phase 3: Coordinate descent refinement
    if elapsed() < deadline - 5:
        h_current = h_best.copy()
        current_val = best_val
        step_sizes = [0.01, 0.05, 0.1, 0.2]
        
        for epoch in range(3):
            if elapsed() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if elapsed() > deadline - 3:
                    break
                old_val = h_current[i]
                for step in step_sizes:
                    for sign in [1, -1]:
                        h_trial = h_current.copy()
                        h_trial[i] = max(0.0, h_current[i] + sign * step)
                        val_trial = compute_metric(h_trial)
                        if val_trial < current_val - 1e-12:
                            h_current = h_trial
                            current_val = val_trial
                            improved = True
                            break
                    if improved:
                        break
                if current_val < best_val:
                    best_val = current_val
                    h_best = h_current.copy()
    
    # Final: ensure nonnegative and return
    h_best = np.maximum(h_best, 0.0)
    if h_best.sum() < 1e-15:
        h_best = np.ones(N)
    
    return list(h_best)
