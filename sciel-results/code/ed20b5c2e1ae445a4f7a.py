import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialize with a Sidon set of size 7
    S = [0, 1, 4, 10, 18, 23, 30]
    h = np.zeros(N)
    for idx in S:
        h[idx] = 1.0
        
    # Normalize to max norm 1
    h = h / np.max(h)
    
    def compute_ratio(h):
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)

    best_h = h.copy()
    best_ratio = compute_ratio(best_h)
    
    # Simple local search
    step_size = 0.1
    for _ in range(10000):
        if time.time() > deadline:
            break
        # Perturb a random element
        i = np.random.randint(N)
        old_val = h[i]
        h[i] = np.clip(h[i] + np.random.uniform(-step_size, step_size), 0, 1)
        # Renormalize
        if np.max(h) > 0:
            h = h / np.max(h)
        
        ratio = compute_ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
        else:
            h[i] = old_val
            if np.max(h) > 0:
                h = h / np.max(h)
            
    return list(best_h)
