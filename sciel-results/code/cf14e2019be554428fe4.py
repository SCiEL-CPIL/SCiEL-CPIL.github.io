import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x."""
    # Use FFT for efficiency
    n = len(x)
    size = 2 * n - 1
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, fft_size)
    conv = np.fft.irfft(X * np.conj(X), fft_size)[:size]
    return conv

def _objective(x, smooth=True, beta=50.0):
    """Compute max|A| / (sum x)^2 with optional smoothing."""
    A = _autoconv(x)
    I = np.sum(x)
    if abs(I) < 1e-12:
        return 1e10
    if smooth:
        # Soft max of |A|
        abs_A = np.abs(A)
        m = np.max(abs_A)
        if m < 1e-12:
            return 0.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_A - m)))) / beta
        return soft_max / (I * I)
    else:
        return np.max(np.abs(A)) / (I * I)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_x = np.ones(N)
    best_val = _objective(best_x, smooth=False)
    
    # Try gradient descent from multiple initializations
    rng = np.random.default_rng(42)
    
    for init in range(20):
        if time.time() > deadline:
            break
        
        if init == 0:
            x = np.ones(N).copy()
        elif init == 1:
            x = rng.standard_normal(N)
        elif init == 2:
            x = rng.choice([-1.0, 1.0], size=N)
        else:
            x = rng.standard_normal(N)
        
        # Normalize so sum is 1 (doesn't change the ratio)
        s = np.sum(x)
        if abs(s) < 1e-12:
            continue
        x = x / s
        
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline:
                break
            
            A = _autoconv(x)
            abs_A = np.abs(A)
            I = np.sum(x)
            
            # Gradient of soft max
            beta = 50.0
            m = np.max(abs_A)
            if m < 1e-12:
                break
            exp_vals = np.exp(beta * (abs_A - m))
            weights = exp_vals / np.sum(exp_vals)
            
            # Gradient of |A[k]| w.r.t. x
            # A[k] = sum_i x[i] * x[i+k]
            # dA[k]/dx[j] = x[j+k] + x[j-k] (with appropriate bounds)
            # This is complex to compute analytically, use numerical or FFT-based
            
            # Actually, let's use the fact that autoconvolution is a quadratic form
            # A = C x where C is a circulant-like matrix
            # For linear autoconvolution, it's not exactly circulant
            
            # Let's use automatic differentiation via finite differences or just
            # use a simpler approach: coordinate descent or subgradient
            
            # Subgradient of max|A|: pick the k that achieves the max
            k_max = np.argmax(abs_A)
            
            # Gradient of A[k_max] w.r.t. x
            # A[k] = sum_i x[i] * x[i+k]
            # dA[k]/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
            grad = np.zeros(N)
            k = k_max
            if k >= 0:
                # A[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
                for i in range(N - k):
                    grad[i] += x[i + k]
                    grad[i + k] += x[i]
            else:
                k_neg = -k
                for i in range(N - k_neg):
                    grad[i] += x[i + k_neg]
                    grad[i + k_neg] += x[i]
            
            # Gradient of 1/I^2 = -2 * (gradient of I) / I^3
            # d(1/I^2)/dx = -2 * (1/I^3) * 1 = -2/I^3
            # So d(max|A|/I^2)/dx = (1/I^2) * grad_A - (2*max|A|/I^3) * 1
            
            val = abs_A[k_max]
            total_grad = (1.0 / (I * I)) * grad - (2.0 * val / (I ** 3)) * np.ones(N)
            
            x -= lr * total_grad
            
            # Re-normalize
            s = np.sum(x)
            if abs(s) < 1e-12:
                break
            x = x / s
        
        val = _objective(x, smooth=False)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x
