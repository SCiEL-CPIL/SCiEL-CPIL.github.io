"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_max(h, k_max=None):
    """Compute the maximum overlap of h with shifted (1-h)."""
    if k_max is None:
        k_max = N - 1
    max_ov = 0.0
    for k in range(-k_max, k_max + 1):
        # Overlap: sum_i h[i] * (1-h)[i+k] for valid indices
        if k >= 0:
            a = h[:N - k]
            b = h[k:]
        else:
            a = h[-k:]
            b = h[:N + k]
        ov = np.sum(a * (1.0 - b))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _logsumexp_max(h, beta=100.0, k_max=None):
    """Smooth approximation of max overlap using log-sum-exp."""
    if k_max is None:
        k_max = N - 1
    overlaps = []
    for k in range(-k_max, k_max + 1):
        if k >= 0:
            a = h[:N - k]
            b = h[k:]
        else:
            a = h[-k:]
            b = h[:N + k]
        ov = np.sum(a * (1.0 - b))
        overlaps.append(ov)
    overlaps = np.array(overlaps)
    max_ov = np.max(overlaps)
    if beta <= 0:
        return max_ov
    shifted = beta * (overlaps - max_ov)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_ov + np.log(log_sum) / beta

def _project(h):
    """Project h onto {0<=h[i]<=1, sum h = N/2}."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if s == 0:
        return np.full(N, 0.5)
    # Adjust to make sum = N/2
    target = N / 2.0
    diff = target - s
    # Distribute diff proportionally, respecting bounds
    for _ in range(100):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to increase some values
            mask = h < 1.0
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        diff = target - s
    return h

def _gradient_smooth(h, beta=100.0, k_max=None):
    """Compute gradient of the log-sum-exp approximation of max overlap."""
    if k_max is None:
        k_max = N - 1
    overlaps = []
    for k in range(-k_max, k_max + 1):
        if k >= 0:
            a = h[:N - k]
            b = h[k:]
        else:
            a = h[-k:]
            b = h[:N + k]
        ov = np.sum(a * (1.0 - b))
        overlaps.append(ov)
    overlaps = np.array(overlaps)
    max_ov = np.max(overlaps)
    shifted = beta * (overlaps - max_ov)
    exps = np.exp(shifted)
    weights = exps / np.sum(exps)
    
    grad = np.zeros(N)
    for idx, k in enumerate(range(-k_max, k_max + 1)):
        w = weights[idx]
        if w < 1e-12:
            continue
        if k >= 0:
            # ov = sum_i h[i] * (1 - h[i+k]) for i in 0..N-k-1
            # d(ov)/d(h[i]) = (1 - h[i+k]) for i in 0..N-k-1
            # d(ov)/d(h[i+k]) = -h[i] for i in 0..N-k-1
            a = h[:N - k]
            b = h[k:]
            grad[:N - k] += w * (1.0 - b)
            grad[k:] += w * (-a)
        else:
            # k < 0, let j = -k
            # ov = sum_i h[i] * (1 - h[i+k]) where i+k >= 0, i.e., i >= -k
            # = sum_i h[i] * (1 - h[i-j]) for i in j..N-1
            j = -k
            a = h[j:]
            b = h[:N - j]
            grad[j:] += w * (1.0 - b)
            grad[:N - j] += w * (-a)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try multiple initializations
    rng = np.random.default_rng(42)
    
    n_restarts = 0
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            h = np.full(N, 0.5)
        elif restart % 3 == 1:
            # Random with mean 0.5
            h = rng.uniform(0, 1, N)
            h = _project(h)
        elif restart % 3 == 2:
            # Structured: alternating high/low
            h = np.where(np.arange(N) % 2 == 0, 0.7, 0.3)
            h = _project(h)
        else:
            # Smooth variation
            t = np.linspace(0, 1, N)
            h = 0.5 + 0.3 * np.sin(2 * np.pi * t)
            h = _project(h)
        
        # Gradient descent
        lr = 0.01
        beta = 50.0
        prev_score = _logsumexp_max(h, beta=beta)
        
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
            
            grad = _gradient_smooth(h, beta=beta)
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            # Check improvement
            cur_score = _logsumexp_max(h_new, beta=beta)
            if cur_score < prev_score:
                h = h_new
                prev_score = cur_score
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        # Evaluate true max
        true_score = _overlap_max(h)
        if true_score < best_score:
            best_score = true_score
            best_h = h.copy()
        
        n_restarts += 1
    
    # Final polish with higher beta
    if time.time() < deadline - 10.0:
        h = best_h.copy()
        beta = 500.0
        lr = 0.005
        prev_score = _logsumexp_max(h, beta=beta)
        
        for step in range(200):
            if time.time() > deadline - 5.0:
                break
            
            grad = _gradient_smooth(h, beta=beta)
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            cur_score = _logsumexp_max(h_new, beta=beta)
            if cur_score < prev_score:
                h = h_new
                prev_score = cur_score
            else:
                lr *= 0.5
                if lr < 1e-7:
                    break
        
        true_score = _overlap_max(h)
        if true_score < best_score:
            best_score = true_score
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
