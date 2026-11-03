import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA_X = 0.5 / N  # interval width: [-1/4, 1/4] has length 1/2

def compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step heights."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -np.inf
    
    # Discrete L2 norm squared, L1 norm
    l2_sq = np.sum(h**2) * DELTA_X
    l1 = np.sum(h) * DELTA_X
    
    if l1 == 0:
        return -np.inf
    
    # Autoconvolution: (h * h)(t) = integral h(s)h(t-s) ds
    # For step functions on uniform grid, we can compute discrete convolution
    # The convolution of two step functions sampled at N points on [-1/4,1/4]
    # We need to compute the convolution properly.
    
    # Let's use np.convolve for the discrete convolution of the height values
    # The step function has value h[i] on interval [x_i, x_{i+1}]
    # The convolution at a point is the integral. For uniform grid, 
    # (h*h)(x_k) ≈ sum_i h[i] * (overlap of intervals)
    
    # Simpler approach: treat as discrete sequence and use np.convolve
    # The convolution of the step approximation
    conv = np.convolve(h, h, mode='full')
    # The convolution values need to be scaled by delta_x^2 for the integral
    # But since we're taking the ratio, and all terms scale consistently,
    # let's just use the raw convolution max
    conv_inf = np.max(conv) * DELTA_X**2
    
    if conv_inf == 0:
        return -np.inf
    
    return l2_sq / (l1 * conv_inf)

def construct():
    start_time = time.time()
    deadline = 150.0  # 150s limit, use 145s for safety
    
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)
    
    # Try several structured families
    def evaluate(h):
        h = np.clip(h, 0, None)
        r = compute_ratio(h)
        return r if np.isfinite(r) else -np.inf
    
    # Family 1: Single spike at different positions
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        r = evaluate(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Family 2: Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            r = evaluate(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
    
    # Family 3: Triangular
    for center in range(N//4, 3*N//4):
        h = np.zeros(N)
        for i in range(N):
            h[i] = max(0, 1 - abs(i - center) / (N//4))
        r = evaluate(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Family 4: Uniform with varying width
    for width in range(1, N+1):
        for offset in range(0, N - width + 1, max(1, (N-width)//5)):
            h = np.zeros(N)
            h[offset:offset+width] = 1.0
            r = evaluate(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
    
    # Family 5: Cosine bump
    for center in range(N//4, 3*N//4):
        for width in range(N//4, N+1):
            h = np.zeros(N)
            for i in range(N):
                d = abs(i - center) / width
                if d <= 1:
                    h[i] = 0.5 * (1 + np.cos(np.pi * d))
            r = evaluate(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
    
    # Local optimization using Nelder-Mead on the best candidate
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Parameterize as log-space to keep nonnegativity
        def objective(log_h):
            h = np.exp(log_h)
            return -evaluate(h)
        
        x0 = np.log(np.maximum(best_h, 1e-10))
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_ratio:
                best_h = np.exp(res.x)
                best_ratio = -res.fun
        except:
            pass
    
    # Additional random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        np.random.seed(42)
        for _ in range(200):
            if time.time() - start_time > deadline - 5:
                break
            # Random structured perturbation
            h = best_h.copy()
            # Randomly modify a few entries
            for _ in range(3):
                idx = np.random.randint(N)
                h[idx] = max(0, h[idx] + np.random.randn() * 0.5)
            r = evaluate(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h.copy()
    
    return list(best_h)
