import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 115.0

def _overlap_max(h):
    """
    Compute the maximum over all cyclic shifts k of the overlap between h and 1-h shifted by k.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    """
    # h is length N, values in [0,1], sum = N/2
    # Overlap[k] = N/2 - autocorr[k]
    # We want to minimize max(Overlap) = N/2 - min(autocorr)
    
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.real(np.fft.ifft(power))
    
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap

def _lse_overlap(h, beta=10.0):
    """
    Log-Sum-Exp approximation of the max overlap.
    Smoother than raw max for gradient-based optimization.
    """
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.real(np.fft.ifft(power))
    overlaps = N / 2.0 - autocorr
    
    # Scale for numerical stability
    max_ov = np.max(overlaps)
    if max_ov == 0:
        return 0.0
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    return lse

def _project(params):
    """
    Project params to feasible set: h in [0,1], sum(h) = N/2.
    Uses bisection on a scaling factor to ensure exact sum and bounds.
    """
    h = np.clip(params, 0, 1)
    target = N / 2.0
    s = np.sum(h)
    
    if abs(s - target) < 1e-12:
        return h

    # If sum is off, we need to adjust.
    # A simple projection is to shift all values by (target - s)/N.
    # But this might violate bounds.
    # We use a "water-filling" or iterative clipping approach.
    
    h_new = h.copy()
    for _ in range(50):
        s = np.sum(h_new)
        diff = target - s
        if abs(diff) < 1e-12:
            break
        
        # Identify indices that can be adjusted
        if diff > 0:
            # Need to increase sum. Increase values < 1.
            mask = h_new < 1.0 - 1e-12
            if not np.any(mask):
                break
            h_new[mask] += diff / np.sum(mask)
        else:
            # Need to decrease sum. Decrease values > 0.
            mask = h_new > 0.0 + 1e-12
            if not np.any(mask):
                break
            h_new[mask] += diff / np.sum(mask)
        
        h_new = np.clip(h_new, 0, 1)
        
    # Final check
    if abs(np.sum(h_new) - target) > 1e-9:
        # Fallback: simple scaling if bounds allow
        if np.all(h_new > 0) and np.all(h_new < 1):
             h_new *= target / np.sum(h_new)
             h_new = np.clip(h_new, 0, 1)
    
    return h_new

def _construct_impl():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Initializations
    initializations = []
    
    # 1. Constant
    initializations.append(np.full(N, 0.5))
    
    # 2. Low-frequency sinusoids
    for k in [1, 2, 3, 5]:
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, 0.5, 1.0]:
                h_init = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
                h_init = np.clip(h_init, 0, 1)
                h_init = _project(h_init)
                initializations.append(h_init)
    
    # 3. Random
    for _ in range(10):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project(h_rand)
        initializations.append(h_rand)

    # Optimization Loop
    for h0 in initializations:
        if time.time() > deadline - 10:
            break
        
        h = h0.copy()
        
        # Gradient Descent with LSE objective
        lr = 0.05
        beta = 5.0
        
        for step in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient of LSE objective
            # L = max + 1/beta * log(sum(exp(beta*(O_k - max))))
            # dL/dh_i = (1/beta) * sum_k exp(beta*O_k) * dO_k/dh_i / sum_k exp(beta*O_k)
            # dO_k/dh_i = 1 - h[(i+k)%N] - h[(i-k)%N] + 2*h[i]*h[(i+k)%N]? 
            # Wait, O_k = sum_j h_j (1 - h_{j+k}) = sum_j h_j - sum_j h_j h_{j+k}
            # dO_k/dh_i = 1 - h_{i+k} - h_{i-k} (from the two terms where i is involved in the product)
            # Actually:
            # Term 1: sum h_j. d/dh_i = 1.
            # Term 2: - sum h_j h_{j+k}.
            # If j=i, term is -h_i h_{i+k}. d/dh_i = -h_{i+k}.
            # If j+k=i => j=i-k, term is -h_{i-k} h_i. d/dh_i = -h_{i-k}.
            # So dO_k/dh_i = 1 - h_{i+k} - h_{i-k}.
            
            H = np.fft.fft(h)
            power = H * np.conj(H)
            autocorr = np.real(np.fft.ifft(power))
            overlaps = N / 2.0 - autocorr
            
            # Weights for LSE
            max_ov = np.max(overlaps)
            weights = np.exp(beta * (overlaps - max_ov))
            weights /= np.sum(weights)
            
            # Gradient
            grad = np.zeros(N)
            # For each k, grad += weights[k] * (1 - h_{i+k} - h_{i-k})
            # This can be done via convolution/FFT or loops.
            # dO_k/dh vector is 1 - shift(h, k) - shift(h, -k)
            
            # Efficient computation:
            # grad = sum_k weights[k] * (1 - np.roll(h, k) - np.roll(h, -k))
            # grad = sum_k weights[k] - sum_k weights[k]*np.roll(h, k) - sum_k weights[k]*np.roll(h, -k)
            # sum_k weights[k] = 1
            # sum_k weights[k]*np.roll(h, k) is the convolution of h with weights (reversed?)
            
            # Let W_k = weights[k].
            # Conv term 1: c1[i] = sum_k W_k h[(i+k)%N]
            # This is circular cross-correlation of h and W.
            # H_fft = fft(h), W_fft = fft(W)
            # c1 = ifft(H_fft * conj(W_fft))? 
            # Correlation: sum_k h[i+k] W[k] = ifft(fft(h) * conj(fft(W)))
            
            H_fft = np.fft.fft(h)
            W_fft = np.fft.fft(weights)
            
            c1 = np.real(np.fft.ifft(H_fft * np.conj(W_fft)))
            c2 = np.real(np.fft.ifft(H_fft * W_fft)) # sum_k W_k h[(i-k)%N]
            
            grad = 1.0 - c1 - c2
            
            # Update
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            # Check improvement using raw score
            score_new = _overlap_max(h_new)
            score_curr = _overlap_max(h)
            
            if score_new < score_curr:
                h = h_new
                lr = min(lr * 1.05, 0.5)
            else:
                lr *= 0.8
            
            if lr < 1e-7:
                break
        
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Polish with coordinate descent on the best candidate
    if time.time() < deadline - 5:
        h = best_h.copy()
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                # Try small changes
                for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    h_trial = h.copy()
                    h_trial[i] += delta
                    h_trial = _project(h_trial)
                    score_trial = _overlap_max(h_trial)
                    if score_trial < _overlap_max(h) - 1e-10:
                        h = h_trial
                        improved = True
                        break
            if not improved:
                break
        
        score = _overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h
    
    return best_h

def construct():
    """
    Return step heights in [0,1] with sum = N/2.
    """
    return _construct_impl()
