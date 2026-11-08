"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    def compute_value(h):
        h = np.asarray(h, dtype=np.float64)
        if len(h) != N:
            return 0.0
        if np.any(h < 0) or np.any(~np.isfinite(h)):
            return 0.0
        g = np.convolve(h, h)
        m = len(g)
        # y = [0, g[0], ..., g[m-1], 0]
        y = np.zeros(m + 2)
        y[1:m+1] = g
        # spacing w
        w = 1.0 / (m + 1)
        # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        l2 = 0.0
        for i in range(m + 1):
            l2 += (w / 3.0) * (y[i]**2 + y[i]*y[i+1] + y[i+1]**2)
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        if norm_1 * norm_inf == 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)
    
    # Seed value
    seed_h = np.ones(N)
    best_h = seed_h.copy()
    best_val = compute_value(best_h)
    
    # Try various structured patterns
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    tri = np.linspace(1, 0, N)
    candidates.append(tri)
    
    # Inverse triangular
    inv_tri = np.linspace(0, 1, N)
    candidates.append(inv_tri)
    
    # Peak in center
    center = np.zeros(N)
    center[N//2] = 1.0
    candidates.append(center)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 * 2)
    candidates.append(gauss)
    
    # Bimodal
    bimodal = np.exp(-((x - 0.5)**2) * 5) + np.exp(-((x + 0.5)**2) * 5)
    candidates.append(bimodal)
    
    # Step function
    step = np.ones(N)
    step[:N//2] = 0.5
    candidates.append(step)
    
    # Cosine window
    cos_win = np.cos(np.pi * x / 2)**2
    candidates.append(cos_win)
    
    # Sinc-like
    sinc = np.sinc(x)
    sinc = np.clip(sinc, 0, None)
    if np.max(sinc) > 0:
        sinc = sinc / np.max(sinc)
    candidates.append(sinc)
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() > deadline:
            break
        c = np.clip(c, 0, None)
        if np.max(c) > 0:
            c = c / np.max(c)  # normalize to max 1
        val = compute_value(c)
        if val > best_val:
            best_val = val
            best_h = c.copy()
    
    # Local optimization using gradient-free methods
    # Use Nelder-Mead style simplex for local refinement
    def nm_optimize(h0, max_iter=200):
        h = h0.copy()
        # Initialize simplex
        simplex = [h.copy()]
        for i in range(N):
            h2 = h.copy()
            h2[i] += 0.1 * max(1.0, abs(h[i]))
            simplex.append(h2)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Sort simplex by value
            values = [compute_value(s) for s in simplex]
            order = np.argsort(values)[::-1]  # descending
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Reflection
            centroid = np.mean(simplex[1:], axis=0)
            reflected = centroid + 1.0 * (centroid - simplex[0])
            reflected = np.clip(reflected, 0, None)
            ref_val = compute_value(reflected)
            
            if ref_val >= values[1]:
                if ref_val >= values[0]:
                    # Expansion
                    expanded = centroid + 2.0 * (reflected - centroid)
                    expanded = np.clip(expanded, 0, None)
                    exp_val = compute_value(expanded)
                    if exp_val > ref_val:
                        simplex[0] = expanded
                    else:
                        simplex[0] = reflected
                else:
                    simplex[0] = reflected
            else:
                # Contraction
                contracted = centroid + 0.5 * (simplex[0] - centroid)
                contracted = np.clip(contracted, 0, None)
                con_val = compute_value(contracted)
                if con_val >= values[0]:
                    simplex[0] = contracted
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
        
        # Return best
        best_local = max(simplex, key=lambda s: compute_value(s))
        return best_local, compute_value(best_local)
    
    # Run local optimization from best candidate
    if time.time() < deadline - 10:
        h_opt, val_opt = nm_optimize(best_h, max_iter=100)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Try more diverse starting points with local optimization
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        # Random smooth pattern
        freq = np.random.uniform(1, 5, 3)
        phase = np.random.uniform(0, 2*np.pi, 3)
        amp = np.random.uniform(0.5, 1.0, 3)
        x = np.linspace(0, 1, N)
        pattern = np.zeros(N)
        for f, p, a in zip(freq, phase, amp):
            pattern += a * np.sin(2*np.pi*f*x + p)
        pattern = np.clip(pattern, 0, None)
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)
        
        h_opt, val_opt = nm_optimize(pattern, max_iter=50)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
    
    return list(best_h)
# EVOLVE-BLOCK-END
