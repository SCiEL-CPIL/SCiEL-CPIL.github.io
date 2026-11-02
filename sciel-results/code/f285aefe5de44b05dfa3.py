"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max_k (S_k) / sum(h)
    where S_k = sum_i h_i h_{i+k}
    """
    h = np.asarray(h, dtype=np.float64)
    if np.sum(h) < 1e-12:
        return np.inf
        
    # Efficiently compute autocorrelation using FFT
    # Zero-pad to at least 2*N to avoid circular convolution effects
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    # Autocorrelation is the inverse FFT of |H|^2
    # Note: numpy fftshift is not needed if we interpret the output correctly,
    # but rfft returns positive frequencies.
    # The linear autocorrelation r[k] for k in 0..N-1 corresponds to the first N elements
    # of the inverse rfft of |H|^2.
    # Let's verify:
    # x = [a, b, c]
    # H = FFT(x)
    # |H|^2 IFFT gives [a^2+b^2+c^2, ab+bc, ac, bc+ab, a^2+b^2+c^2] ... wait.
    # Linear autocorrelation via FFT:
    # 1. Pad x to length 2N-1 (or next power of 2)
    # 2. FFT
    # 3. Multiply by conjugate
    # 4. IFFT
    # 5. Take first N elements (for lags 0 to N-1)
    
    auto = np.fft.irfft(H * np.conj(H), n=n_fft)
    # The first N elements of 'auto' are the autocorrelations for lags 0 to N-1.
    # Let's double check with a small example mentally.
    # x = [1, 0, 0]
    # H = [1, 1, 1] (if N=3, n_fft=4? No, 2*3-1=5, next pow2=8)
    # |H|^2 = [1,1,1,1,1,1,1,1]
    # IFFT -> [1, 0, 0, 0, 0, 0, 0, 0]
    # First 3: [1, 0, 0]. Correct.
    
    # x = [1, 1, 0]
    # H = FFT([1,1,0,0,0,0,0,0])
    # |H|^2 IFFT -> [2, 1, 0, -1, 0, -1, 0, 1]? No, autocorrelation of real signal is symmetric.
    # r[0] = 1^2+1^2 = 2
    # r[1] = 1*1 = 1
    # r[2] = 0
    # First 3: [2, 1, 0]. Correct.
    
    S = auto[:N]
    H_sum = np.sum(h)
    return np.max(S) / H_sum

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Try simple analytic shapes
    candidates = []
    
    # Uniform
    h_uniform = np.ones(N)
    candidates.append(h_uniform)
    
    # Triangle
    idx = np.arange(N)
    center = (N - 1) / 2.0
    h_tri = np.clip(1 - np.abs(idx - center) / center, 0, 1)
    if np.sum(h_tri) > 0:
        candidates.append(h_tri)
        
    # Gaussian
    for sigma in [N/4, N/8, N/16]:
        h_gauss = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
        candidates.append(h_gauss)

    best_score = np.inf
    best_h = None
    
    for h in candidates:
        # Normalize to sum 1 for optimization, but score is scale invariant
        h_norm = h / np.sum(h)
        score = compute_score(h_norm)
        if score < best_score:
            best_score = score
            best_h = h_norm.copy()

    # 2. Convex Optimization
    # Minimize max_k S_k subject to sum(h) = 1, h >= 0
    # Using a smooth approximation of the max function: Log-Sum-Exp
    
    def smooth_max_and_grad(h, beta=50.0):
        # h is assumed to sum to 1 (or close to it)
        # Compute S
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n=n_fft)
        auto = np.fft.irfft(H * np.conj(H), n=n_fft)
        S = auto[:N]
        
        # Smooth max
        m = np.max(S)
        exp_S = np.exp(beta * (S - m))
        sum_exp = np.sum(exp_S)
        val = m + np.log(sum_exp) / beta
        
        # Gradient
        # grad_val = (1/beta) * (sum(exp_S * grad_S) / sum_exp) + grad_m
        # grad_m = grad_{k*} where k* = argmax(S)
        # This is expensive to compute analytically for FFT-based S.
        # However, we can use the fact that grad_S_k = h shifted.
        # But doing this in the gradient of the LSE is complex.
        
        # Alternative: Use SLSQP with finite differences? N=600 might be slow.
        # Alternative: Use a simpler surrogate or just projected gradient descent on the smooth max.
        
        # Let's try projected gradient descent on the smooth max.
        # We need the gradient of S_k with respect to h.
        # S_k = sum_i h_i h_{i+k}
        # dS_k/dh_j = h_{j+k} (if j+k < N) + h_{j-k} (if j-k >= 0)
        
        # To get grad of LSE(S), we need weights w_k = exp(beta(S_k - m)) / sum_exp
        # grad LSE = sum_k w_k * grad S_k
        
        weights = exp_S / sum_exp
        
        grad_S = np.zeros_like(h)
        # We can compute this efficiently?
        # grad_S_j = sum_k w_k (h_{j+k} + h_{j-k})
        # This is a convolution of w with h?
        # Let's just compute it directly for now, N=600 is small enough for O(N^2) in a few iterations?
        # 600^2 = 360,000. Very fast.
        
        # Actually, let's just do the O(N^2) loop.
        for k in range(N):
            if weights[k] < 1e-8:
                continue
            w = weights[k]
            # Add w * h_{j+k} to grad_S_j
            # This means grad_S[:N-k] += w * h[k:]
            # And grad_S[k:] += w * h[:N-k]
            grad_S[:N-k] += w * h[k:]
            grad_S[k:] += w * h[:N-k]
            
        return val, grad_S

    # Optimization Loop
    h_opt = best_h.copy()
    h_opt = h_opt / np.sum(h_opt) # Ensure sum 1
    
    step_size = 0.01
    beta = 20.0
    
    for i in range(500):
        if time.time() > deadline:
            break
            
        val, grad = smooth_max_and_grad(h_opt, beta)
        
        # Projected Gradient Descent
        # Update h
        h_new = h_opt - step_size * grad
        
        # Projection: h >= 0, sum(h) = 1
        # This is not trivial with two constraints.
        # Simple approach: Clip to 0, then normalize.
        h_new = np.clip(h_new, 0, None)
        s = np.sum(h_new)
        if s > 1e-12:
            h_new /= s
        else:
            h_new = best_h.copy()
            
        # Check improvement
        new_score = compute_score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            
        # Adaptive step size
        if new_score > best_score * 1.01:
            step_size *= 0.5
        else:
            step_size = min(step_size * 1.1, 0.1)
            
        h_opt = h_new

    # Return the best found
    # The problem asks for a step function. The evaluator measures the raw value.
    # "score is the upper bound on C1".
    # "The evaluator measures the raw value (lower is better)."
    # The seed program returns [1.0] * N.
    # If we return [1.0] * N, what is the score?
    # S_0 = sum(1*1) = 600.
    # sum(h) = 600.
    # Score = 600/600 = 1.0.
    
    # If we return h_normalized (sum=1), then S_0 = 1. Score = 1/1 = 1.0.
    # The score is scale invariant.
    # So we can return best_h directly.
    
    return list(best_h)

# EVOLVE-BLOCK-END
