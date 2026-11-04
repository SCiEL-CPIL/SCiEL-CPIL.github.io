import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _autoconv(x):
    """Compute full autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2 for k != 0."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    if len(c) > 1:
        max_sidelobe = np.max(np.abs(c[1:]))
    else:
        max_sidelobe = 0.0
    return max_sidelobe / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Initialize with constant vector
    x = np.ones(N)
    best_val = _eval_ratio(x)
    best_result = x.copy()
    
    # Gradient Descent on x
    # Objective: minimize max_{k!=0} |c_k| / S^2
    # Use soft-max for differentiability
    
    def _soft_max_abs(c, k=10.0):
        abs_c = np.abs(c[1:])
        # Add small epsilon to avoid log(0)
        return np.log(np.sum(np.exp(k * abs_c))) / k
    
    def _objective(x):
        c = _autoconv(x)
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        return _soft_max_abs(c) / (s * s)
    
    x_curr = np.ones(N)
    np.random.seed(42)
    x_curr += 0.1 * np.random.randn(N)
    
    lr = 0.01
    steps = 200 # Limit steps to ensure time
    
    for i in range(steps):
        if time.time() > deadline:
            break
        
        val = _eval_ratio(x_curr)
        if val < best_val:
            best_val = val
            best_result = x_curr.copy()
            
        # Compute gradient numerically? No, use analytical.
        # J = M / S^2
        # dJ/dx = (1/S^2) dM/dx - (2M/S^3) dS/dx
        # dS/dx = 1
        # M = max |c_k| (approx soft-max)
        # dM/dx is complex.
        
        # Let's just use a simple heuristic: 
        # If c_k is large positive, reduce x_i x_{i+k}.
        # This is tricky.
        
        # Alternative: Random restarts with spectral shaping.
        # We found that flat spectrum is bad for linear conv.
        # What if we use a "chirp" in the linear sense?
        # x_n = cos(pi * n^2 / N)
        
    # Try a few specific structured sequences
    # 1. Chirp
    t = np.arange(N)
    x_chirp = np.cos(np.pi * t**2 / N)
    val = _eval_ratio(x_chirp)
    if val < best_val:
        best_val = val
        best_result = x_chirp.copy()
        
    # 2. M-Sequence like (random binary with sum 0, plus constant)
    for restart in range(10):
        if time.time() > deadline:
            break
        y = np.random.choice([-1, 1], N)
        # Ensure sum is not 0
        s_y = np.sum(y)
        if abs(s_y) < 1:
            y[0] = -y[0]
            s_y = np.sum(y)
            
        # Try different constants A
        for A in [0.5, 1.0, 2.0, 5.0, 10.0]:
            x_new = A * np.ones(N) + y
            val = _eval_ratio(x_new)
            if val < best_val:
                best_val = val
                best_result = x_new.copy()
                
    return best_result
