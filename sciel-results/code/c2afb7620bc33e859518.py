import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    sup_norm = np.max(conv)
    if sup_norm == 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    
    def score(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return -1e10
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(conv)
        if sup_norm < 1e-15:
            return -1e10
        return l2_sq / (l1 * sup_norm)
    
    def score_smooth(h, beta=50.0):
        """Smooth surrogate using log-sum-exp for the max."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return -1e10
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        # Smooth max via log-sum-exp
        m = np.max(conv)
        if m < 1e-15:
            return -1e10
        log_sum_exp = m + np.log(np.sum(np.exp((conv - m) * beta)) / len(conv)) / beta
        sup_norm_smooth = np.exp(log_sum_exp)
        if sup_norm_smooth < 1e-15:
            return -1e10
        return l2_sq / (l1 * sup_norm_smooth)
    
    def gradient_ascent(h0, max_steps=5000, lr_init=0.01, beta=50.0):
        """Gradient ascent using finite differences on smooth score."""
        h = np.array(h0, dtype=float)
        h = np.maximum(h, 0)
        # Normalize to avoid scale issues
        s = np.sum(h)
        if s > 0:
            h = h / s
        
        best_h = h.copy()
        best_val = score_smooth(h, beta)
        exact_best = score(h)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient via finite differences
            grad = np.zeros_like(h)
            eps = 1e-5
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] = max(h[i] + eps, 0)
                h_minus = h.copy()
                h_minus[i] = max(h[i] - eps, 0)
                fp = score_smooth(h_plus, beta)
                fm = score_smooth(h_minus, beta)
                grad[i] = (fp - fm) / (2 * eps)
            
            # Normalize gradient
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break
            
            # Adaptive learning rate
            lr = lr_init / (1 + step * 0.01)
            h_new = h + lr * grad / gnorm
            h_new = np.maximum(h_new, 0)
            
            # Renormalize
            s_new = np.sum(h_new)
            if s_new > 0:
                h_new = h_new / s_new
            
            v_new = score_smooth(h_new, beta)
            if v_new > best_val:
                h = h_new
                best_val = v_new
                best_h = h.copy()
                exact_best = score(h)
            else:
                lr_init *= 0.95
        
        return best_h, exact_best
    
    def try_structural_patterns():
        candidates = []
        
        # Single spike
        h = np.zeros(N)
        h[0] = 1.0
        candidates.append(h)
        
        # Two adjacent spikes with various ratios
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:
            h = np.zeros(N)
            h[0] = 1.0
            h[1] = ratio
            candidates.append(h)
        
        # Three consecutive spikes
        for a, b, c in [(1, 0.5, 0.25), (1, 0.7, 0.3), (1, 1, 0.5), (1, 0.8, 0.6), (1, 0.6, 0.4)]:
            h = np.zeros(N)
            h[0] = a
            h[1] = b
            h[2] = c
            candidates.append(h)
        
        # Cosine window
        for width in [3, 5, 7, 10]:
            h = np.zeros(N)
            for i in range(width):
                h[i] = np.cos(np.pi * i / (width + 1))
            candidates.append(h)
        
        # Exponential decay
        for decay in [0.5, 0.7, 0.8, 0.9, 0.95]:
            h = np.zeros(N)
            for i in range(N):
                h[i] = decay ** i
            candidates.append(h)
        
        # Gaussian-like
        for sigma in [1.0, 1.5, 2.0, 3.0]:
            h = np.zeros(N)
            for i in range(N):
                h[i] = np.exp(-i**2 / (2 * sigma**2))
            candidates.append(h)
        
        # Sparse random patterns
        rng = np.random.RandomState(123)
        for _ in range(200):
            h = np.zeros(N)
            k = rng.randint(1, 10)
            indices = rng.choice(N, size=k, replace=False)
            for idx in indices:
                h[idx] = rng.random()
            if np.sum(h) > 0:
                candidates.append(h)
        
        return candidates
    
    # Try structural patterns first
    best_h = None
    best_val = -1
    
    for h in try_structural_patterns():
        v = score(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
        if time.time() > deadline - 30:
            break
    
    # Run gradient ascent from best structural pattern
    if best_h is not None:
        h_opt, v_opt = gradient_ascent(best_h, max_steps=3000, lr_init=0.05, beta=50.0)
        if v_opt > best_val:
            best_val = v_opt
            best_h = h_opt
    
    # Run gradient ascent from several random starts
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random sparse initialization
        h_init = np.zeros(N)
        k = rng.randint(1, 15)
        indices = rng.choice(N, size=k, replace=False)
        for idx in indices:
            h_init[idx] = rng.random()
        
        if np.sum(h_init) < 1e-10:
            continue
        
        h_opt, v_opt = gradient_ascent(h_init, max_steps=2000, lr_init=0.03, beta=40.0)
        if v_opt > best_val:
            best_val = v_opt
            best_h = h_opt
    
    # Final polish with smaller steps
    if best_h is not None and time.time() < deadline - 5:
        h_opt, v_opt = gradient_ascent(best_h, max_steps=2000, lr_init=0.01, beta=100.0)
        if v_opt > best_val:
            best_val = v_opt
            best_h = h_opt
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
