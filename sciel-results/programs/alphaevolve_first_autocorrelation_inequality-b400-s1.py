"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_and_grad(h, beta=100.0):
    """
    Compute smooth approximation of the score and its gradient.
    Uses log-sum-exp for smooth max.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    # Smooth max using log-sum-exp
    max_conv = np.max(conv)
    # Soft-max: sum(exp(beta*(conv - max_conv))) / beta
    shifted = conv - max_conv
    logsumexp = np.log(np.sum(np.exp(beta * shifted))) / beta
    smooth_max = max_conv + logsumexp
    
    score = smooth_max / (s * s)
    
    # Gradient of smooth_max w.r.t. h
    # d(smooth_max)/d(conv[k]) = exp(beta*(conv[k] - max_conv)) / sum(exp(beta*(conv[j] - max_conv)))
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # d(conv[k])/d(h[i]) = h[i+k] + h[i-k] (with appropriate bounds)
    # conv[k] = sum_j h[j] h[j+k] for k >= 0
    # For linear convolution of length 2N-1:
    # conv[k] = sum_{j} h[j] h[j+k] where j ranges so that both indices valid
    
    grad_smooth_max = np.zeros_like(h)
    for k in range(len(conv)):
        # conv[k] = sum_{j} h[j] * h[j+k] (for k >= 0, with offset)
        # Actually, np.convolve(h,h,'full')[k] = sum_{j} h[j] h[k-j]
        # Let me compute gradient properly
        pass
    
    # Simpler: use finite differences or direct formula
    # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] (symmetric)
    # Let's just use the fact that conv = h * h (linear convolution)
    # grad of conv w.r.t h: for each output k, d(conv[k])/d(h[i]) = h[k-i] + h[i-k]
    
    # Build gradient
    for i in range(N):
        # Contribution to conv[k] from h[i]
        # conv[k] = sum_j h[j] h[k-j]
        # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] (if indices valid)
        # But this is expensive to do for all k
        pass
    
    # Alternative: use the formula that gradient of sum_k weights[k]*conv[k] w.r.t h[i]
    # = sum_k weights[k] * (h[k-i] + h[i-k])
    
    # Let's compute it via FFT or direct
    # For now, use a simpler approach: coordinate-wise gradient
    grad_score = np.zeros_like(h)
    
    # d(score)/d(h[i]) = (1/(s^2)) * d(smooth_max)/d(h[i]) - 2*smooth_max/s^3 * 1
    # = (1/(s^2)) * d(smooth_max)/d(h[i]) - 2*score/s
    
    # Compute d(smooth_max)/d(h[i])
    # smooth_max = max_conv + (1/beta)*log(sum(exp(beta*(conv[k]-max_conv))))
    # d(smooth_max)/d(h[i]) = sum_k weights[k] * d(conv[k])/d(h[i])
    
    # d(conv[k])/d(h[i]) where conv = np.convolve(h, h, 'full')
    # conv[k] = sum_{j=max(0,k-(N-1))}^{min(N-1,k)} h[j] * h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] (if 0<=k-i<N) + h[i-k] (if 0<=i-k<N)
    
    # This is O(N^2) which is fine for N=600
    for i in range(N):
        grad_i = 0.0
        for k in range(len(conv)):
            j1 = k - i
            j2 = i - k
            contrib = 0.0
            if 0 <= j1 < N:
                contrib += h[j1]
            if 0 <= j2 < N:
                contrib += h[j2]
            grad_i += weights[k] * contrib
        grad_score[i] = grad_i / (s * s) - 2 * score / s
    
    return score, grad_score

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Conservative safety margin
    
    # Initial candidate: uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    x = np.linspace(-0.25, 0.25, N)
    
    # Evaluate parametric candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Cosine window
    candidates.append(0.5 * (1 + np.cos(4 * np.pi * x)))
    
    # 3. Gaussian
    candidates.append(np.exp(-x**2 / (2*0.1**2)))
    
    # 4. Power bump: (1-t^2)^p
    for p in [0.5, 1, 2, 3, 4, 5, 10]:
        t = x / 0.25
        h_p = np.maximum(0, 1 - t**2) ** p
        candidates.append(h_p)
    
    # 5. Two-humped
    for center in [0.1, 0.15, 0.2]:
        for w in [0.01, 0.02, 0.05, 0.1]:
            h_2h = np.exp(-(x-center)**2/w) + np.exp(-(x+center)**2/w)
            candidates.append(h_2h)
    
    # Evaluate initial candidates
    for h in candidates:
        if time.time() > deadline:
            break
        h = np.maximum(0, h)
        if np.sum(h) == 0:
            continue
        score = compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Gradient descent with smooth surrogate
    h = best_h.copy()
    beta = 50.0  # Temperature for smooth max
    
    # Normalize to sum=1 for stability
    h = h / np.sum(h)
    
    lr = 0.01
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        score, grad = compute_score_and_grad(h, beta)
        
        if np.sum(np.abs(grad)) < 1e-10:
            break
        
        # Gradient step with projection to nonnegative orthant
        h_new = h - lr * grad
        h_new = np.maximum(0, h_new)
        
        # Renormalize
        s = np.sum(h_new)
        if s > 0:
            h_new = h_new / s
        
        new_score = compute_score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # Reduce learning rate
            lr *= 0.9
        
        # Anneal beta
        if iteration % 20 == 0:
            beta *= 1.5
    
    # Final polish: coordinate descent
    h = best_h.copy()
    for cd_iter in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = h[i]
            # Try small perturbations
            for factor in [0.5, 0.75, 1.25, 1.5]:
                h[i] = old_val * factor
                if h[i] < 0:
                    h[i] = 0
                score = compute_score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
                    improved = True
            h[i] = old_val  # Restore
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
