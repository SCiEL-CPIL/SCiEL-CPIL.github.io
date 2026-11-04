import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _soft_max(x, t=100.0):
    """Differentiable approximation of max(x)."""
    # Using log-sum-exp for stability
    m = np.max(x)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp((x - m) / t))) * t

def _objective(x, t=100.0):
    """
    Minimize log(||h||_1) + log(||h*h||_inf) subject to ||h||_2 = 1.
    Since the objective is scale invariant, we fix ||h||_2 = 1.
    The score is 1 / (||h||_1 * ||h*h||_inf).
    Minimizing the log product is equivalent to maximizing the score.
    """
    h = x
    if np.any(h < -1e-12):
        return 1e10
    h = np.maximum(h, 0)
    
    l1 = np.sum(h)
    if l1 == 0:
        return 1e10
    
    conv = _autoconvolution(h)
    # We want to minimize log(l1 * max(conv))
    # Note: conv is non-negative for non-negative h
    max_conv = np.max(conv)
    if max_conv == 0:
        return 1e10
        
    # Use soft max for differentiability of the max term
    # Actually, for L-BFGS, we can just use the max if we are okay with non-smoothness, 
    # but soft max is better.
    # However, the gradient of soft max is simple.
    # Let's stick to a standard smooth approximation or just use the max if the optimizer handles it.
    # L-BFGS-B handles non-smoothness poorly. Let's use soft max.
    
    # Soft max of conv
    # To avoid overflow in exp, we shift
    # But conv values are around 1. t=100 is large enough to be close to max.
    sm = _soft_max(conv, t)
    
    return np.log(l1) + np.log(sm)

def _objective_grad(x, t=100.0):
    """
    Gradient of log(||h||_1) + log(soft_max(h*h)).
    """
    h = x
    if np.any(h < -1e-12):
        return np.zeros(N)
    h = np.maximum(h, 0)
    
    l1 = np.sum(h)
    if l1 == 0:
        return np.zeros(N)
        
    conv = _autoconvolution(h)
    max_conv = np.max(conv)
    if max_conv == 0:
        return np.zeros(N)
        
    # Gradient of log(l1)
    grad_l1 = np.ones(N) / l1
    
    # Gradient of log(soft_max(conv))
    # soft_max(conv) = exp(sum(log(exp(conv_i / t)))) * t ... wait.
    # Let S = sum(exp(conv_i / t)). soft_max = t * log(S).
    # d/dh log(soft_max) = d/dh log(t * log(S)) = d/dh log(S) = (1/S) * sum(exp(conv_i/t) * d(conv_i)/dh)
    
    # d(conv_i)/dh = 2 * (h shifted by i)
    # This is computationally expensive to do analytically for every i.
    # Alternative: use numerical gradient or accept non-smooth max.
    # Given the time limit, let's try using the standard max and see if L-BFGS-B works with numerical gradients.
    # Or, simpler: just use the max and let scipy handle it with finite differences.
    # But finite differences on 50 dimensions is slow.
    
    # Let's try a different approach:
    # Maximize Score directly? No, non-convex.
    
    # Let's use the analytical gradient for the soft max.
    # conv_k = sum_j h_j h_{j+k} (for full conv index k)
    # d(conv_k)/dh_m = h_{m-k} + h_{m+k} (with bounds checks)
    
    # This is O(N^2) per gradient evaluation. N=50, so 2500 ops. Very fast.
    
    # Calculate weights for soft max
    shifted = conv / t
    exp_shifted = np.exp(shifted - np.max(shifted))
    S = np.sum(exp_shifted)
    
    # d(log(S))/dh_m = (1/S) * sum_k exp_shifted_k * d(conv_k)/dh_m
    # d(conv_k)/dh_m = h_{m-k} + h_{m+k}
    # So sum_k exp_shifted_k * (h_{m-k} + h_{m+k})
    # This is a convolution of exp_shifted with h, plus another one.
    
    # Let E_k = exp_shifted_k
    # Term 1: sum_k E_k h_{m-k} -> Convolution of E and h
    # Term 2: sum_k E_k h_{m+k} -> Convolution of E and h (reversed/shifted)
    
    # Since E is the weights of the soft max of the autoconvolution, and h is the signal.
    # We can compute this via FFT or direct sum. Direct sum is fine for N=50.
    
    grad_sm = np.zeros(N)
    for m in range(N):
        val = 0.0
        for k in range(2*N - 1):
            # d(conv_k)/dh_m
            # conv_k = sum_j h_j h_{j+k}
            # If m = j, term is h_{m+k}
            # If m = j+k, term is h_{m-k}
            
            # Contribution from j=m: h_{m+k} if m+k < N
            if m + k < N:
                val += E[k] * h[m+k]
            # Contribution from j+k=m => j=m-k: h_{m-k} if m-k >= 0
            if m - k >= 0:
                val += E[k] * h[m-k]
                
        grad_sm[m] = (val / S) * (1/t) # Chain rule: d(soft_max)/dconv * dconv/dh * 1/soft_max
        # Wait. soft_max = t * log(S).
        # d(soft_max)/dh = t * (1/S) * sum E_k dconv_k/dh
        # d(log(soft_max))/dh = (1/soft_max) * d(soft_max)/dh = (1/(t log S)) * t (1/S) sum ...
        # = (1/(S log S)) * sum E_k dconv_k/dh
        
    # Let's re-verify the derivative.
    # f = log(soft_max(conv))
    # soft_max = t * log( sum exp(conv_i/t) )
    # df/dh = (1/soft_max) * t * (1/sum_exp) * sum exp(conv_i/t) * dconv_i/dh
    # df/dh = (1/(t log S)) * (1/S) * sum exp(conv_i/t) * dconv_i/dh
    
    denom = t * np.log(S)
    if denom == 0:
        return grad_l1
        
    grad_sm = grad_sm / denom
    
    return grad_l1 + grad_sm

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Initialize with a few diverse starts
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N))
    
    # 2. Centered Gaussian
    x = np.arange(N)
    starts.append(np.exp(-((x - N//2)**2) / (2 * (N/4)**2)))
    
    # 3. Random sparse
    for _ in range(5):
        h = np.zeros(N)
        idx = np.random.choice(N, 5, replace=False)
        h[idx] = 1.0
        starts.append(h)
        
    # 4. Random dense
    for _ in range(5):
        h = np.random.uniform(0.1, 1.0, N)
        starts.append(h)
        
    # 5. Bimodal
    for _ in range(2):
        h = np.zeros(N)
        h[N//4] = 1.0
        h[3*N//4] = 1.0
        # Add some noise
        h += np.random.uniform(0, 0.2, N)
        starts.append(h)
        
    for s in starts:
        if time.time() > deadline - 10:
            break
            
        # Normalize to L2=1
        norm = np.linalg.norm(s)
        if norm == 0:
            continue
        x0 = s / norm
        
        # Optimize
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
        except Exception:
            continue
            
        h_opt = res.x
        if np.any(h_opt < 0):
            h_opt = np.maximum(h_opt, 0)
            
        # Check if valid
        score = _score(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # Final polish: check if we can improve by scaling? 
    # No, scale invariant.
    
    # One more attempt: use the best found as a start for a more aggressive optimization
    if best_score > 0 and time.time() < deadline - 5:
        x0 = best_h / np.linalg.norm(best_h)
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            h_opt = np.maximum(res.x, 0)
            score = _score(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass

    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
