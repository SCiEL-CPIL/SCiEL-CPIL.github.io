import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Initial candidates
    candidates = []
    
    # 1. Random binary
    h1 = np.random.choice([-1, 1], N)
    candidates.append(h1)
    
    # 2. Random real
    h2 = np.random.randn(N)
    candidates.append(h2)
    
    # 3. Chirp
    t = np.linspace(0, 1, N)
    h3 = np.cos(2 * np.pi * (t**2) * N)
    candidates.append(h3)
    
    # 4. Golay-like (random signs with some structure)
    h4 = np.ones(N)
    for i in range(N):
        if np.random.rand() < 0.5:
            h4[i] = -1
    candidates.append(h4)

    best_h = None
    best_score = float('inf')
    
    # Normalize and evaluate initial candidates
    for h in candidates:
        h = h.astype(float)
        if np.sum(h) == 0:
            h += 1e-9
        h = h / np.sum(h)
        score = evaluate(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    if best_h is None:
        return np.ones(N)
        
    h = best_h.copy()
    current_score = best_score
    
    # Local Search
    # We want to minimize max(conv(h,h)) subject to sum(h)=1
    # Objective: max_k c_k
    
    # Precompute convolution
    conv = np.convolve(h, h)
    max_idx = np.argmax(conv)
    max_val = conv[max_idx]
    
    delta = 0.1
    no_improve = 0
    
    while time.time() < deadline:
        # Pick two indices
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
            
        # Perturb
        # Keep sum constant: h[i] += delta, h[j] -= delta
        h_new = h.copy()
        h_new[i] += delta
        h_new[j] -= delta
        
        # Evaluate
        # Optimization: Only recompute conv if necessary?
        # For N=400, conv is fast.
        conv_new = np.convolve(h_new, h_new)
        max_val_new = np.max(conv_new)
        
        # We minimize max_val_new
        if max_val_new < max_val:
            h = h_new
            conv = conv_new
            max_val = max_val_new
            max_idx = np.argmax(conv)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 1000:
                delta *= 0.9
                no_improve = 0
                if delta < 1e-6:
                    break
            
    # Final normalization check (should be 1, but floating point errors)
    h = h / np.sum(h)
    
    return h

def evaluate(h):
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    return abs(2 * N * max_c / (s ** 2))
