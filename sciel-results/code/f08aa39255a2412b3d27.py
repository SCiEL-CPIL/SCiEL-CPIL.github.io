import time
import numpy as np
from scipy import optimize

N = 600
M = N // 2
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1

def _fft_convolve(h):
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def _compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    return np.max(conv) / (s * s)

def _smooth_score_and_grad(h, beta=100.0):
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros(N)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    if m <= 0:
        return 0.0, np.zeros(N)
    
    # LogSumExp
    exp_terms = np.exp((conv - m) * beta)
    sum_exp = np.sum(exp_terms)
    lse = m + np.log(sum_exp) / beta
    
    # Weights for gradient
    w = exp_terms / sum_exp
    
    # Gradient of LSE w.r.t h
    # d(LSE)/d(h[j]) = sum_k w[k] * d(conv[k])/d(h[j])
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-(k-N+1)]
    # This is equivalent to:
    # g[j] = sum_k w[k] h[k-j] + sum_k w[k] h[j-(k-N+1)]
    
    # Term 1: sum_k w[k] h[k-j] is convolution of w and h, but aligned differently.
    # Let's use FFT for the gradient calculation to be safe and fast.
    # Actually, we can just compute the convolution of w with h and w with reversed h.
    
    # conv_w_h = w * h
    # Note: conv[k] = sum_i h[i] h[k-i].
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k+N-1] (using 0-indexed, careful with bounds)
    
    # Let's use a direct FFT approach for the gradient of the quadratic form if possible?
    # No, w is data-dependent.
    
    # Efficient calculation:
    # g1[j] = sum_k w[k] h[k-j]
    # This is the correlation of w and h?
    # Let's just use numpy correlate or FFT.
    
    # Pad w and h to _FFT_SIZE
    w_pad = np.zeros(_FFT_SIZE)
    w_pad[:len(conv)] = w
    
    h_pad = np.zeros(_FFT_SIZE)
    h_pad[:N] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    # Convolution w * h
    # In frequency domain: W * H
    # But we need specific alignments.
    
    # Alternative: Just use the fact that for symmetric h and w (w is symmetric if h is symmetric and we are at peak? No, w is not necessarily symmetric if peak is not at center? 
    # The peak of autoconvolution of a symmetric function is at the center.
    # So w is symmetric around the center of conv.
    # h is symmetric around the center of h.
    
    # Let's compute g using the definition:
    # g[j] = sum_k w[k] (h[k-j] + h[j-(k-N+1)])
    
    # Since N=600 is small, we can do this with FFT.
    # Term 1: sum_k w[k] h[k-j]
    # This is the cross-correlation of w and h?
    # Let c1 = np.correlate(w, h, mode='full')? No, w is length 2N-1, h is N.
    
    # Let's stick to the simple O(N^2) if N is small? 600^2 = 360k.
    # 360k ops per gradient step.
    # 500 steps * 360k = 180M ops.
    # In Python, this might be too slow (1-2 seconds per step?).
    # We need FFT.
    
    # Let's use the property:
    # grad_LSE = 2 * IFFT(FFT(w) * FFT(h))?
    # Check:
    # If LSE was sum(conv), then grad = 2 h * h.
    # With weights w, it is like a weighted sum.
    # It is exactly the convolution of w with h, plus w with flipped h?
    
    # Let's try:
    # g = np.zeros(N)
    # w_conv_h = np.fft.irfft(W * H, n=_FFT_SIZE)
    # w_flip_conv_h = np.fft.irfft(W * np.conj(H), n=_FFT_SIZE) # Correlation?
    
    # Actually, for real h:
    # d/dh (sum w_k h_k * h) = w * h + flip(w) * h
    # Here w_k is the weight for conv[k].
    # conv[k] = sum_i h_i h_{k-i}
    # d conv[k] / d h_j = h_{k-j} + h_{j-(k-N+1)}
    
    # Let's just compute the gradient using the formula:
    # g[j] = sum_k w[k] h[k-j] + sum_k w[k] h[j-(k-N+1)]
    
    # We can compute the first sum using FFT.
    # sum_k w[k] h[k-j]
    # Let u = w, v = h.
    # This is (u * v)[j] if we align them correctly?
    # Standard conv: (u*v)[j] = sum_k u[k] v[j-k].
    # We have sum_k w[k] h[k-j].
    # Let m = k-j => k = m+j.
    # sum_m w[m+j] h[m].
    # This is correlation.
    
    # Let's use scipy.signal.fftconvolve or similar?
    # Or just:
    # g1 = np.fft.irfft(np.fft.rfft(w_pad) * np.fft.rfft(h_pad), n=_FFT_SIZE)
    # This gives (w * h).
    # (w * h)[j] = sum_k w[k] h[j-k].
    # We want sum_k w[k] h[k-j].
    # This is (w * flip(h))[j]?
    # Let flip(h)[i] = h[N-1-i].
    # No, let's just use the symmetry.
    # If h is symmetric, and w is symmetric (peak at center), then the gradient is symmetric.
    
    # Let's assume h is symmetric (we enforce it).
    # Then conv is symmetric. Peak is at center. w is symmetric.
    # Then grad is symmetric.
    # We only need to compute half the gradient.
    
    # Let's use a simple loop for the gradient if FFT alignment is tricky, but optimize with NumPy broadcasting?
    # 600x600 matrix multiplication?
    # w (600, 1) * h (1, 600)? No.
    
    # Let's try the FFT approach again.
    # g[j] = sum_k w[k] h[k-j]
    # This is the cross-correlation of w and h at lag j?
    # Correlation R[j] = sum_k w[k+j] h[k].
    # If we reverse w, we get convolution.
    
    # Let's just use:
    # g1 = np.fft.irfft(np.fft.rfft(w_pad) * np.fft.rfft(np.flip(h_pad)), n=_FFT_SIZE)
    # g2 = np.fft.irfft(np.fft.rfft(np.flip(w_pad)) * np.fft.rfft(h_pad), n=_FFT_SIZE)
    # This is getting confusing.
    
    # SAFE FALLBACK:
    # Since N=600, we can use a matrix multiplication approach with a precomputed matrix?
    # No, w changes.
    
    # Let's use the fact that N is small enough for a vectorized NumPy operation.
    # k = np.arange(_CONV_LEN)
    # j = np.arange(N)
    # We need h[k-j] and h[j-(k-N+1)].
    # Indices must be valid.
    
    # Let's just use L-BFGS with finite differences if analytic gradient is too complex?
    # 300 variables. FD requires 300 evaluations.
    # 300 * (FFT cost) ~ 300 * 10us = 3ms per gradient.
    # 500 steps * 3ms = 1.5s.
    # This is perfectly acceptable!
    
    # So, we will use `scipy.optimize.minimize` with `method='L-BFGS-B'` and `jac=None` (finite differences).
    # To save time, we will optimize only the first M=300 variables (symmetric half).
    
    return lse / (s * s), None

def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 130.0

    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    # Symmetric optimization
    # h_full[0:M] = x
    # h_full[M:N] = x[::-1]
    
    def expand(x):
        return np.concatenate([x, x[::-1]])
        
    def objective_and_grad(x):
        h = expand(x)
        s = np.sum(h)
        if s == 0:
            return 0.0, np.zeros(M)
        conv = _fft_convolve(h)
        score = np.max(conv) / (s * s)
        return score, None
        
    def objective(x):
        return objective_and_grad(x)[0]
        
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(M))
    
    # 2. Gaussian
    for sigma in [50, 100, 150, 200]:
        x = np.arange(M) - (M - 1) / 2
        # Map to full range for Gaussian center?
        # Center of h is at (N-1)/2 = 299.5.
        # Center of x (0..299) is 149.5.
        # Distance from center in full space:
        # i in 0..299 -> full index i.
        # dist = abs(i - 299.5)
        d = np.abs(np.arange(M) - (N - 1) / 2.0)
        h_gauss = np.exp(-(d ** 2) / (2 * sigma ** 2))
        initializations.append(h_gauss)
        
    # 3. Cosine Bump
    for L in [100, 150, 200, 250]:
        d = np.abs(np.arange(M) - (N - 1) / 2.0)
        h_cos = np.where(d < L, np.cos(np.pi * d / (2 * L)), 0)
        initializations.append(h_cos)
        
    # 4. Triangle
    for L in [150, 200, 250]:
        d = np.abs(np.arange(M) - (N - 1) / 2.0)
        h_tri = np.maximum(0, 1 - d / L)
        initializations.append(h_tri)

    for x0 in initializations:
        if time.time() > deadline:
            break
        
        # Normalize x0 to sum to 1 to help optimizer?
        s0 = np.sum(x0)
        if s0 > 0:
            x0 = x0 / s0 * 10 # Scale up to avoid 0 issues, optimizer will handle scale?
            # Actually, the score is scale invariant.
            # We can fix sum(h) = 1 to reduce dimensionality?
            # No, L-BFGS can handle it.
            # But fixing sum=1 makes the landscape better conditioned?
            # Let's just optimize as is.
            
        try:
            res = optimize.minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                bounds=[(0, None)] * M,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            h_cand = expand(res.x)
            h_cand = np.maximum(0, h_cand)
            
            sc = _compute_score(h_cand)
            
            if sc < best_score:
                best_score = sc
                best_h = h_cand.copy()
                
        except Exception:
            pass
            
    return best_h.tolist()
