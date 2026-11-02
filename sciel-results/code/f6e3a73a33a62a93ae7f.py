"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _max_overlap_value(h):
    """Return the maximum overlap value."""
    corrs = _compute_overlaps(h)
    return np.max(corrs)

def _soft_max_overlap(h, beta=100.0):
    """Soft-max of overlap values using log-sum-exp for smoothness."""
    corrs = _compute_overlaps(h)
    max_c = np.max(corrs)
    if max_c < -1e15:
        return 0.0, np.zeros(len(h))
    exps = np.exp(beta * (corrs - max_c))
    log_sum = np.log(np.sum(exps)) / beta + max_c
    # Gradient: d/dh_j of soft_max = (1/N) sum_k (2/N) h[i+k] * softmax_weight_k
    # R_k = (2/N) sum_i h[i]h[i+k]
    # dR_k/dh_j = (2/N) * (h[j+k] if j < n-k else 0) + (2/N) * (h[j-k] if j >= k else 0)
    n = len(h)
    weights = exps / np.sum(exps)
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * beta
        if w < 1e-15:
            continue
        # Contribution to gradient from R_k
        grad[:n-k] += w * (2.0/n) * h[k:]
        grad[k:] += w * (2.0/n) * h[:n-k]
    return log_sum, grad

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(30):
        if abs(diff) < 1e-13:
            break
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.sum(mask) == 0:
            # All at bounds, do a small uniform adjustment
            h += diff / len(h)
            h = np.clip(h, 0, 1)
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
        diff = target_sum - np.sum(h)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute among elements not at strict bounds
        interior = (h > 1e-12) & (h < 1 - 1e-12)
        if np.sum(interior) > 0:
            h[interior] += diff / np.sum(interior)
            h = np.clip(h, 0, 1)
        else:
            h += diff / len(h)
            h = np.clip(h, 0, 1)
    # Ensure exact sum with tiny correction
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-14:
        h[0] += diff
        h = np.clip(h, 0, 1)
    return h

def _score(h):
    """Score is the max overlap (lower is better)."""
    return _max_overlap_value(h)

def construct():
    n = N
    target_sum = n / 2.0
    
    np.random.seed(42)
    
    best_h = np.full(n, 0.5)
    best_score = _score(best_h)
    
    # Strategy 1: Smooth gradient descent with soft-max surrogate
    # Start from various initializations and run projected gradient descent
    
    initializations = []
    
    # Constant
    initializations.append(np.full(n, 0.5))
    
    # Sine wave patterns
    for freq in [1, 2, 3, 5, 10, 20, 25, 50]:
        for amp in [0.1, 0.2, 0.3, 0.4, 0.49]:
            t = np.arange(n)
            h_init = 0.5 + amp * np.sin(2 * np.pi * freq * t / n)
            h_init = np.clip(h_init, 0, 1)
            h_init = _project_to_simplex(h_init, target_sum)
            initializations.append(h_init.copy())
    
    # Random smooth patterns
    for trial in range(20):
        # Generate smooth random h via low-frequency Fourier
        coeffs = np.zeros(n, dtype=complex)
        for k in range(1, 8):
            amp = np.random.uniform(0, 0.3)
            phase = np.random.uniform(0, 2 * np.pi)
            coeffs[k] = amp * np.exp(1j * phase)
        h_init = np.full(n, 0.5)
        for k in range(1, 8):
            if np.abs(coeffs[k]) > 1e-10:
                h_init += np.real(coeffs[k] * np.exp(2j * np.pi * k * np.arange(n) / n))
        h_init = np.clip(h_init, 0, 1)
        h_init = _project_to_simplex(h_init, target_sum)
        initializations.append(h_init.copy())
    
    # Bang-bang patterns
    for trial in range(15):
        h_bb = np.random.choice([0, 1], size=n).astype(float)
        h_bb = _project_to_simplex(h_bb, target_sum)
        initializations.append(h_bb.copy())
    
    # Run gradient descent from each initialization
    for init_idx, h0 in enumerate(initializations):
        if _time_remaining() < 3.0:
            break
        
        h = h0.copy()
        current_score = _score(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Adaptive beta schedule for soft-max
        beta_start = 50.0
        beta_end = 500.0
        
        lr = 0.01
        patience = 0
        max_patience = 50
        
        for step in range(300):
            if _time_remaining() < 1.5:
                break
            
            # Schedule beta
            progress = step / 300.0
            beta = beta_start + (beta_end - beta_start) * progress**2
            
            soft_val, grad = _soft_max_overlap(h, beta=beta)
            
            # Project gradient descent step
            h_new = h - lr * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_score = _score(h_new)
            
            if new_score < current_score - 1e-10:
                h = h_new
                current_score = new_score
                patience = 0
                lr = min(lr * 1.02, 0.05)
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                patience += 1
                lr *= 0.95
                if patience > max_patience:
                    break
        
        # Final polish with exact max
        if _time_remaining() > 2.0:
            h_polish = h.copy()
            for step in range(100):
                if _time_remaining() < 1.0:
                    break
                
                corrs = _compute_overlaps(h_polish)
                max_val = np.max(corrs)
                active = corrs >= max_val - 1e-8
                
                grad = np.zeros(n)
                for k in range(1, n):
                    if not active[k-1]:
                        continue
                    grad[:n-k] += (2.0/n) * h_polish[k:]
                    grad[k:] += (2.0/n) * h_polish[:n-k]
                
                norm = np.linalg.norm(grad)
                if norm < 1e-15:
                    break
                grad /= norm
                
                h_new = h_polish - 0.002 * grad
                h_new = _project_to_simplex(h_new, target_sum)
                
                new_score = _score(h_new)
                if new_score < _score(h_polish):
                    h_polish = h_new
    
    # Strategy 2: Local search on the best solution found
    if _time_remaining() > 10.0:
        h_ls = best_h.copy()
        current_score = _score(h_ls)
        
        # Coordinate descent / local perturbation
        for epoch in range(5):
            if _time_remaining() < 5.0:
                break
            
            improved = False
            for i in range(n):
                if _time_remaining() < 1.0:
                    break
                
                # Try perturbing h[i]
                for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                    h_trial = h_ls.copy()
                    h_trial[i] += delta
                    h_trial = _project_to_simplex(h_trial, target_sum)
                    
                    new_score = _score(h_trial)
                    if new_score < current_score - 1e-10:
                        h_ls = h_trial
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_h = h_ls.copy()
                        break
            
            if not improved:
                break
    
    # Strategy 3: Try structured patterns with specific properties
    if _time_remaining() > 15.0:
        # Try h[i] = 0.5 + a * cos(2*pi*i/n) + b * cos(4*pi*i/n)
        for a in np.linspace(-0.4, 0.4, 9):
            for b in np.linspace(-0.3, 0.3, 7):
                if _time_remaining() < 2.0:
                    break
                t = np.arange(n)
                h_struct = 0.5 + a * np.cos(2 * np.pi * t / n) + b * np.cos(4 * np.pi * t / n)
                h_struct = np.clip(h_struct, 0, 1)
                h_struct = _project_to_simplex(h_struct, target_sum)
                
                s = _score(h_struct)
                if s < best_score:
                    best_score = s
                    best_h = h_struct.copy()
    
    # Ensure final result is valid
    best_h = _project_to_simplex(best_h, target_sum)
    
    return best_h
# EVOLVE-BLOCK-END
