import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial best
    best_h = np.ones(N)
    best_val = 2.0
    
    # Helper to compute value
    def calc_val(h):
        s = np.sum(h)
        if s == 0:
            return float('inf')
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        return abs(2 * N * max_c / (s * s))
    
    # Try a few random strategies
    # Strategy 1: Random with mean 1
    for _ in range(50):
        if time.time() > deadline:
            break
        h = 1.0 + 0.1 * np.random.randn(N)
        v = calc_val(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
            
    # Strategy 2: Smoothed cosine waves
    for freq in range(1, 10):
        if time.time() > deadline:
            break
        t = np.linspace(0, 2 * np.pi * freq, N)
        for amp in [0.1, 0.5, 1.0, 2.0]:
            h = 1.0 + amp * np.cos(t)
            v = calc_val(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
                
    # Strategy 3: Local refinement of best_h using coordinate descent
    # This might be slow if done naively, so limit iterations
    if time.time() < deadline - 10:
        h = best_h.copy()
        v = best_val
        step_size = 0.01
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Pick a random index
            i = np.random.randint(N)
            # Try changing h[i]
            old_h_i = h[i]
            # Simple line search or fixed step
            # To keep it fast, just try a small perturbation
            h[i] = old_h_i + step_size * np.random.randn()
            new_v = calc_val(h)
            if new_v < v:
                v = new_v
                best_val = v
                best_h = h.copy()
            else:
                h[i] = old_h_i
                # Decay step size if stuck?
                # step_size *= 0.99
                
    return best_h
