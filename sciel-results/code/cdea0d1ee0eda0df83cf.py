"""Spectral shaping approach for the first autocorrelation inequality."""
import time
import numpy as np

N = 600


def _autoconv_peak(h):
    """Return the peak of the linear autoconvolution of h (nonnegative)."""
    # Linear autoconvolution: (h * h)(k) = sum_j h[j] h[k-j]
    # For nonnegative h, this is equivalent to the circular autoconvolution
    # when we zero-pad sufficiently. Use FFT for efficiency.
    n = len(h)
    m = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_len = 1
    while fft_len < m:
        fft_len *= 2
    h_ext = np.zeros(fft_len)
    h_ext[:n] = h
    H = np.fft.fft(h_ext)
    conv = np.fft.ifft(H * H).real[:m]
    return np.max(conv)


def _score(h):
    """Normalized maximum of autoconvolution: peak / (sum h)^2 * N^2
    The score is the upper bound on C1, which is the normalized peak.
    For step functions on N equal intervals, the autoconvolution peak
    divided by (integral of h)^2 times some normalization gives C1.
    
    Actually, let's think about this more carefully.
    The step function is h[i] on interval i of [-1/4, 1/4], each of width 1/(4N).
    The integral is sum(h[i]) * (1/(4N)).
    The autoconvolution (f*f)(x) = integral f(y)f(x-y)dy.
    For a step function, the autoconvolution at the peak is maximized.
    
    The normalized maximum: max_x (f*f)(x) / (integral f)^2
    This is dimensionless and gives C1.
    
    Let's compute it properly:
    - integral = sum(h) * dw where dw = 1/(4N)
    - autoconv at discrete points: for the step function, the autoconvolution
      is piecewise linear. The peak occurs at some point.
    
    Actually, for the discrete approximation, if we think of h as values on
    grid points, the autoconvolution peak in the continuous sense for a step
    function can be computed via the discrete autoconvolution scaled appropriately.
    
    Let me use a simpler approach: the score is max(autoconv) / (sum(h))^2 * (N^2)
    or something similar. Let me just use the ratio that makes the uniform case
    give score 1.0.
    
    For uniform h = [1, 1, ..., 1] (N ones):
    autoconv peak = N (at the center, sum of N*1*1)
    Actually for linear autoconv of N ones: the peak is N (each term is 1*1, N terms)
    Wait no. The linear autoconv of [1,1,...,1] (N elements) has length 2N-1.
    At lag 0: N (all N pairs align)
    At lag 1: N-1
    ...
    So peak = N.
    
    sum(h) = N
    So max(autoconv) / (sum(h))^2 = N / N^2 = 1/N
    
    For the normalized score to be 1.0 for uniform, we need:
    score = max(autoconv) * N / (sum(h))^2 = N * N / N^2 = 1. Yes!
    
    So score = N * max_autoconv_peak / (sum(h))^2
    """
    s = np.sum(h)
    if s == 0:
        return float('inf')
    peak = _autoconv_peak(h)
    return N * peak / (s * s)


def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)  # Should be 1.0
    
    # Spectral shaping: work in Fourier domain
    # For nonnegative h, the DFT magnitudes can be shaped
    # Start with uniform spectrum
    
    h = np.ones(N)
    current_score = best_score
    
    # Gradient descent on the nonnegative heights
    # Use a simple approach: iteratively adjust heights based on autoconv profile
    
    for step in range(2000):
        if time.time() > deadline:
            break
        
        s = np.sum(h)
        if s == 0:
            h = np.ones(N)
            continue
        
        # Compute autoconvolution
        n = len(h)
        m = 2 * n - 1
        fft_len = 1
        while fft_len < m:
            fft_len *= 2
        h_ext = np.zeros(fft_len)
        h_ext[:n] = h
        H = np.fft.fft(h_ext)
        conv = np.fft.ifft(H * H).real[:m]
        
        peak = np.max(conv)
        score = N * peak / (s * s)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Gradient computation: we want to minimize peak/autoconv
        # The autoconv at lag k is sum_j h[j] h[k-j]
        # d(peak)/d(h[i]) is tricky, but we can use a heuristic:
        # Reduce heights where they contribute most to the peak, increase elsewhere
        
        # Find where the peak is
        peak_idx = np.argmax(conv)
        
        # The peak is at conv[peak_idx] = sum_j h[j] h[peak_idx - j] (with proper indexing)
        # For the linear autoconv, conv[k] = sum_{j} h[j] * h[k-j] for valid j
        
        # Simple heuristic: proportional update
        # h_new[i] = h[i] * (1 - lr * contribution_to_peak[i])
        # where contribution_to_peak[i] measures how much h[i] contributes to the peak
        
        # For peak at index p in autoconv:
        # conv[p] = sum_{j=0}^{N-1} h[j] * h[p-j] (with h out of range = 0)
        # d conv[p] / d h[i] = sum_j (h[p-j] * delta(i-j) + h[j] * delta(i-(p-j)))
        #                      = h[p-i] + h[p-i]  (if both in range)
        #                      = 2*h[p-i] (roughly, if p-i in [0,N-1])
        
        # So the gradient w.r.t. h[i] for the peak term is ~ h[p-i]
        # But we need to account for normalization by s^2
        
        lr = 0.01
        p = peak_idx
        
        # Contribution vector
        contrib = np.zeros(N)
        for i in range(N):
            j = p - i
            if 0 <= j < N:
                contrib[i] = h[j]
        
        # Normalize contribution
        if np.max(contrib) > 0:
            contrib_norm = contrib / np.max(contrib)
        
        # Update: reduce where contribution is high, increase where low
        # But keep nonnegative
        h_new = h - lr * contrib_norm
        h_new = np.maximum(h_new, 0.0)
        
        # Also add a small random perturbation for exploration
        if step % 50 == 0:
            h_new += 0.01 * np.random.randn(N)
            h_new = np.maximum(h_new, 0.0)
        
        # Normalize to keep same sum (optional, but helps with scaling)
        s_new = np.sum(h_new)
        if s_new > 0:
            h_new = h_new * (s / s_new)
        
        h = h_new
        current_score = score
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Random nonnegative initialization
            h_rand = np.abs(np.random.randn(N)) + 0.1
            h_rand = h_rand / np.max(h_rand)
            
            for step in range(500):
                if time.time() > deadline:
                    break
                
                s = np.sum(h_rand)
                if s == 0:
                    break
                
                n = len(h_rand)
                m = 2 * n - 1
                fft_len = 1
                while fft_len < m:
                    fft_len *= 2
                h_ext = np.zeros(fft_len)
                h_ext[:n] = h_rand
                H = np.fft.fft(h_ext)
                conv = np.fft.ifft(H * H).real[:m]
                
                peak = np.max(conv)
                score = N * peak / (s * s)
                
                if score < best_score:
                    best_score = score
                    best_h = h_rand.copy()
                
                peak_idx = np.argmax(conv)
                p = peak_idx
                
                contrib = np.zeros(N)
                for i in range(N):
                    j = p - i
                    if 0 <= j < N:
                        contrib[i] = h_rand[j]
                
                if np.max(contrib) > 0:
                    contrib_norm = contrib / np.max(contrib)
                else:
                    contrib_norm = np.zeros(N)
                
                lr = 0.01
                h_new = h_rand - lr * contrib_norm
                h_new = np.maximum(h_new, 0.0)
                
                s_new = np.sum(h_new)
                if s_new > 0:
                    h_new = h_new * (s / s_new)
                
                h_rand = h_new
    
    # Return the best solution found
    return list(best_h)
