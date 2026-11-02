import numpy as np
import time

N = 100

def _autocorr(h):
    """Compute circular autocorrelation: ac[k] = sum_i h[i] * h[(i+k) % N]"""
    # Use FFT for speed
    H = np.fft.fft(h)
    ac = np.fft.ifft(H * np.conj(H)).real
    return ac

def _min_autocorr(h):
    ac = _autocorr(h)
    # Exclude k=0 (which is sum h^2)
    return np.min(ac[1:])

def _soft_min_autocorr(h, eps=0.01):
    ac = _autocorr(h)
    ac = ac[1:]  # exclude k=0
    # soft min: -log-sum-exp(-ac/eps)
    return -eps * np.log(np.sum(np.exp(-ac / eps)))

def _project(h, target_sum=None):
    """Project h onto [0,1]^N with sum = target_sum"""
    if target_sum is None:
        target_sum = N / 2.0
    h = np.clip(h, 0, 1)
    # Water-filling to fix sum
    diff = h.sum() - target_sum
    if abs(diff) < 1e-12:
        return h
    # Sort to find threshold for water-filling
    idx = np.argsort(h)
    if diff > 0:
        # Need to decrease sum: reduce the largest elements
        # Water fill from top
        h_sorted = h[idx]
        # We want to subtract from the top elements
        # Find how many top elements to adjust
        n_adj = 0
        cum = 0.0
        for i in range(N-1, -1, -1):
            n_adj += 1
            val = h[idx[i]]
            # If we reduce this element by delta, max reduction is val
            if cum + val >= diff:
                # This element gets partially reduced
                remaining = diff - cum
                h[idx[i]] = val - remaining
                break
            else:
                h[idx[i]] = 0.0
                cum += val
        else:
            # All reduced to 0, but still need more - shouldn't happen if feasible
            pass
    else:
        # Need to increase sum: increase the smallest elements
        h_sorted = h[idx]
        n_adj = 0
        cum = 0.0
        for i in range(N):
            n_adj += 1
            val = h[idx[i]]
            # Max increase for this element is 1 - val
            max_inc = 1.0 - val
            if cum + max_inc >= -diff:
                remaining = -diff - cum
                h[idx[i]] = val + remaining
                break
            else:
                h[idx[i]] = 1.0
                cum += max_inc
        else:
            pass
    return h

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Initialize with Barker-like structure
    # Barker sequences have low autocorrelation sidelobes
    # For even N, use a balanced sequence
    # Start with alternating pattern modified
    h = np.full(N, 0.5)
    
    # Try a few initializations
    best_h = h.copy()
    best_score = _min_autocorr(h)
    
    # Barker-inspired: use a sequence with low sidelobes
    # For N=100, we can use a composite Barker or just start random
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initializations
    inits = [
        np.full(N, 0.5),
        (rng.random(N) < 0.5).astype(float) * 0.5 + 0.25,  # random binary-like
        np.tile([0.0, 1.0], N // 2),  # alternating
        np.tile([1.0, 0.0], N // 2),
    ]
    
    for init in inits:
        h = _project(init.copy())
        cur_score = _min_autocorr(h)
        if cur_score > best_score:
            best_score = cur_score
            best_h = h.copy()
    
    h = best_h.copy()
    
    # Subgradient ascent on min autocorrelation
    # The min autocorrelation is non-smooth, use subgradient
    eps = 0.05
    lr = 0.01
    n_iter = 2000
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        ac = _autocorr(h)
        ac_sidelobes = ac[1:]
        
        # Find which shift(s) achieve the minimum
        min_val = np.min(ac_sidelobes)
        min_idx = np.argmin(ac_sidelobes)
        
        # Subgradient of min_k ac[k] w.r.t. h
        # ac[k] = sum_i h[i] h[(i+k)%N]
        # d ac[k] / d h[j] = h[(j-k)%N] + h[(j+k)%N] (for circular)
        # For the minimizing k, subgradient is the gradient of ac[k]
        k = min_idx
        # d ac[k]/d h[j] = h[(j-k)%N] + h[(j+k)%N]
        grad = np.zeros(N)
        for j in range(N):
            grad[j] = h[(j - k) % N] + h[(j + k) % N]
        
        # Ascent: h += lr * grad, then project
        h_new = h + lr * grad
        h_new = _project(h_new)
        
        new_score = _min_autocorr(h_new)
        if new_score > _min_autocorr(h) - 1e-9:
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.95
        
        if it % 100 == 0:
            if new_score > best_score:
                best_score = new_score
                best_h = h.copy()
    
    # Final refinement with smaller eps
    eps = 0.01
    lr = 0.005
    for it in range(1000):
        if time.time() > deadline:
            break
        
        ac = _autocorr(h)
        ac_sidelobes = ac[1:]
        min_val = np.min(ac_sidelobes)
        min_idx = np.argmin(ac_sidelobes)
        k = min_idx
        
        grad = np.zeros(N)
        for j in range(N):
            grad[j] = h[(j - k) % N] + h[(j + k) % N]
        
        h_new = h + lr * grad
        h_new = _project(h_new)
        
        new_score = _min_autocorr(h_new)
        if new_score > _min_autocorr(h) - 1e-9:
            h = h_new
        else:
            lr *= 0.95
        
        if new_score > best_score:
            best_score = new_score
            best_h = h.copy()
    
    h = best_h
    # Ensure exact sum
    h = _project(h)
    
    # Verify sum is exactly N/2
    diff = h.sum() - N / 2.0
    if abs(diff) > 1e-10:
        # Tiny adjustment
        h += diff / N
        h = np.clip(h, 0, 1)
    
    return h
