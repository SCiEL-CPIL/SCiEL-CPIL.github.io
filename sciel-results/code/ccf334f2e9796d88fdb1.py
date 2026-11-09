import time
import numpy as np
from scipy.fft import fft, ifft

N = 600

def _autoconv(h):
    """Compute full linear autoconvolution of h using FFT."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = fft(h, n=size)
    ac = ifft(H * H, n=size).real
    return ac[:2*n-1]

def _score(h):
    """Compute the normalized maximum of autoconvolution (the value to minimize)."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    h_norm = h / s
    ac = _autoconv(h_norm)
    return np.max(ac) * N

def construct():
    """
    Structural change from 'return uniform' to an active optimization loop.
    
    The uniform distribution has value 1.0. However, by shaping the distribution
    to have a 'ring' or 'hollow' structure (low center, high sides), we can reduce
    the peak of the autoconvolution. The peak at lag 0 is sum(h_i^2).
    For a fixed sum, sum(h_i^2) is minimized when h is uniform. 
    Wait, if we make h non-uniform, sum(h_i^2) increases.
    So the peak at lag 0 increases.
    
    BUT, the autoconvolution peak might not be at lag 0?
    For non-negative h, the autoconvolution is a positive function.
    Is the maximum always at lag 0?
    (f*f)(x) = integral f(t) f(x-t) dt.
    If f is non-negative, is (f*f)(0) the max?
    Not necessarily. If f is two separated bumps, the cross-correlation between them
    might create a peak at the distance between them, which could be higher than
    the self-correlation if the bumps are narrow and high?
    No, for non-negative functions, the autocorrelation is maximized at lag 0
    IF the function is unimodal and symmetric? No, that's not a general theorem.
    
    Let's test: h = [1, 0, 1]. Sum = 2. h_norm = [0.5, 0, 0.5].
    Autoconv:
    lag -1: 0
    lag 0: 0.25 + 0.25 = 0.5
    lag 1: 0
    lag 2: 0.25
    Max is 0.5. Value = 0.5 * 3 = 1.5. Uniform: 1.0.
    
    What if h = [1, 1, 0]? Sum=2. h_norm=[0.5, 0.5, 0].
    Autoconv:
    lag -1: 0
    lag 0: 0.5
    lag 1: 0.25
    Max 0.5. Value 1.5.
    
    It seems uniform is locally optimal or globally optimal for non-negative?
    
    Let's look at Inspiration 3 (score 1.1457). It claims to improve.
    How?
    Maybe the "normalized maximum" is defined differently?
    "minimize the normalized maximum of its autoconvolution"
    Usually normalized by L2 norm or L1 norm.
    If normalized by L1 (sum), we did that.
    If normalized by L2 (sqrt(sum h^2))?
    Score = max(ac) / (sum h^2).
    Uniform: max(ac) = 1/N (with sum=1). sum h^2 = 1/N. Ratio = 1.
    Same.
    
    Let's try a different approach: **Spectral Shaping**.
    We want to minimize max(ac).
    ac = IFFT(|FFT(h)|^2).
    We want to flatten ac.
    This is equivalent to making the power spectrum |FFT(h)|^2 as flat as possible?
    No, flat power spectrum -> white noise -> ac is a spike (Dirac delta).
    We want the *time-domain* ac to be flat.
    This means the *frequency-domain* spectrum should be concentrated?
    If ac is constant, its FFT is a spike at frequency 0.
    So |FFT(h)|^2 should be a spike at 0.
    This means h should be a constant (in frequency), which means h is a delta (in time)?
    No.
    
    Let's use a heuristic: **Iterative Peak Reduction**.
    1. Start with uniform h.
    2. Compute ac.
    3. Find the lags where ac is high.
    4. Adjust h to reduce contributions to those lags.
    
    Alternatively, use **Gradient Descent** on a smooth approximation of max(ac).
    Loss = sum(exp(beta * ac_k) / beta).
    Minimize this subject to h >= 0, sum(h) = 1.
    
    Let's implement Adam on h with this loss.
    """
    
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Initialize with uniform
    h = np.ones(N) / N
    
    # Hyperparameters
    lr = 0.01
    beta = 10.0  # Sharpness of max approximation
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    
    best_h = h.copy()
    best_score = _score(h)
    
    while time.time() < deadline:
        t += 1
        
        # Compute autoconv
        ac = _autoconv(h)
        
        # Compute loss gradient (smooth max)
        # L = (1/beta) * sum(exp(beta * ac_k))
        # dL/dac_k = exp(beta * ac_k)
        # dac_k/dh_i = h_{k-i} (for appropriate indices)
        
        # Efficient gradient computation via convolution
        # grad_h = conv(exp(beta*ac), h)
        # Note: ac is full length 2N-1. h is length N.
        # We need to be careful with indices.
        
        e_ac = np.exp(beta * ac)
        
        # Gradient of ac_k w.r.t h_i is h_{k-i} if 0 <= k-i < N else 0.
        # So grad_h_i = sum_k e_ac_k * (d ac_k / d h_i)
        # This is a correlation/convolution.
        
        # Let's use FFT for efficiency
        size = 1
        while size < 2 * N - 1:
            size *= 2
        
        E_ac = fft(e_ac, n=size)
        H = fft(h, n=size)
        
        # The gradient in frequency domain
        # This is tricky because ac is linear convolution.
        # Let's just do it in time domain for simplicity if N=600 is small enough?
        # 600 * 1200 = 720,000 ops. Fast.
        
        grad = np.zeros(N)
        for i in range(N):
            # h_i contributes to ac_k where k-i is in [0, N-1]
            # So k is in [i, i+N-1]
            # Term is h_{k-i} = h_j where j = k-i
            # So for fixed i, we sum over k from i to i+N-1: e_ac_k * h_{k-i}
            # Let j = k-i. j goes from 0 to N-1.
            # Sum over j: e_ac_{i+j} * h_j
            
            # This is a sliding window dot product
            # We can precompute e_ac padded?
            pass
        
        # Faster: 
        # grad_i = sum_{j=0}^{N-1} h_j * e_ac_{i+j}
        # This is a cross-correlation of h and e_ac.
        
        # Using FFT
        # cross_corr(h, e_ac)[i] = sum_j h_j e_ac_{i+j}
        # This is equivalent to convolution of h and reverse(e_ac)?
        
        # Let's just use the loop, N=600 is small.
        # Optimize the inner loop with NumPy
        # We can compute this as a batch of dot products?
        # Or use scipy.signal.correlate?
        
        # Let's use a simple loop with NumPy slicing
        # e_ac is length 2N-1.
        # For i in 0..N-1:
        # grad[i] = dot(h, e_ac[i:i+N])
        
        # This can be vectorized if we construct a matrix? 
        # Or use stride tricks?
        # For N=600, a loop of 600 dot products of size 600 is fine.
        
        for i in range(N):
            grad[i] = np.dot(h, e_ac[i:i+N])
            
        # Update h with Adam
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        h_new = h - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project to simplex: h >= 0, sum(h) = 1
        # This is the Euclidean projection onto the simplex.
        # Algorithm:
        # 1. Sort h_new in descending order
        # 2. Find rho such that sum_{i in S} (h_new_i - tau) = 1
        # where S is the set of indices where h_new_i > tau.
        
        u = np.sort(h_new)[::-1]
        cssv = np.cumsum(u)
        cond = u - (cssv - 1) / np.arange(1, N + 1) > 0
        rho = cond.sum()
        if rho == 0:
            h_new = np.zeros(N)
            h_new[0] = 1
        else:
            tau = (cssv[rho - 1] - 1) / rho
            h_new = np.maximum(h_new - tau, 0)
            
        # Normalize just in case
        s = np.sum(h_new)
        if s > 0:
            h_new /= s
        else:
            h_new = np.ones(N) / N
            
        h = h_new
        
        # Check score
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return [float(x) for x in best_h]

BET: {mechanism: "Using smooth max approximation (softmax-like) with Adam optimizer on the time-domain heights allows the distribution to shape itself to reduce the autoconvolution peak, potentially finding a non-uniform local minimum that is better than uniform if the uniform is not globally optimal for this specific metric definition.", q: 0.3, delta: [0.0, 0.2]}
