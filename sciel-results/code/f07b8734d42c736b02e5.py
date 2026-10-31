import numpy as np
import time

N = 600
DEADLINE = 145  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def compute_conv_max(h):
        """Compute max of linear autoconvolution of h."""
        c = np.convolve(h, h, mode='full')
        return np.max(c)
    
    def metric(h):
        """Normalized metric: max_conv / (sum_h)^2, scale-invariant."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        return compute_conv_max(h) / (s * s)
    
    # Log-sum-exp smooth approximation of max
    def smooth_max(c, temp=0.01):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))
    
    def smooth_metric(h, temp=0.01):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        c = np.convolve(h, h, mode='full')
        return smooth_max(c, temp) / (s * s)
    
    # Evaluate uniform baseline
    h_uniform = np.ones(N)
    best_val = metric(h_uniform)
    best_h = h_uniform.copy()
    
    # Phase 1: Try structured candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper (reduces edge effects in convolution)
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    # Sinc-like (flat top with ringing)
    n = np.arange(N)
    center = N / 2 - 0.5
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)
    
    # Gaussian
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Raised cosine (Blackman window)
    n = np.arange(N)
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning window
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top window
    flat_top = 0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) + 0.277263158 * np.cos(4*np.pi*n/(N-1)) + 0.083578947 * np.cos(6*np.pi*n/(N-1)) + 0.006947368 * np.cos(8*np.pi*n/(N-1))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau: high at edges, low in center (to reduce center peak)
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse: low at edges, high in center
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    # Evaluate candidates
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Gradient-based optimization with smooth max
    # Parameterize h = softmax(x) * N to keep sum ~ N (scale doesn't matter for normalized metric)
    # Actually, since metric is scale-invariant, we can just use h = exp(x) and normalize
    
    from scipy.optimize import minimize
    
    def objective(x):
        if elapsed() > DEADLINE - 10:
            return 1e10
        # Convert to positive heights
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=0.01)
    
    # Start from best candidate
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)  # normalize
    
    result = None
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        if result.success or result.fun < best_val:
            h_opt = np.exp(result.x - np.max(result.x))
            v = metric(h_opt)
            if v < best_val:
                best_val = v
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Phase 3: Coordinate descent with exact max (not smooth)
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    # Precompute for fast conv updates
    # When we change h[i], the convolution changes in a structured way
    # c[k] = sum_j h[j] h[j+k], changing h[i] affects c[i-j] for all j and c[j-i] for all j
    
    def update_conv(h, conv, i, old_val, new_val):
        """Update convolution when h[i] changes from old_val to new_val."""
        delta = new_val - old_val
        if abs(delta) < 1e-15:
            return conv
        # c[k] = sum_j h[j] h[j+k]
        # When h[i] changes by delta:
        # c[i-j] += h[j]*delta for all valid j (where i-j is valid lag)
        # c[j-i] += h[j]*delta for all valid j
        # Actually: c[k] = sum_j h[j] * h[j+k]
        # If h[i] changes, it contributes to c[k] when i = j (h[i]*h[i+k]) or i = j+k (h[j]*h[i])
        # So c[k] gets delta*h[i+k] from the first term (j=i) and delta*h[i-k] from second term (j=i-k)
        # Wait, let me be more careful.
        # c[k] = sum_{j: 0<=j<N, 0<=j+k<2N-1... } h[j]*h[j+k] for linear conv of length 2N-1
        # Actually np.convolve(h,h,'full') gives c[k] = sum_j h[j]*h[j+k] where k ranges from -(N-1) to N-1
        # Let's index c from 0 to 2N-2, where c[k] = sum_j h[j]*h[j+k-N+1]... 
        # This is getting complicated. Let me just recompute for small changes.
        return None  # Skip this optimization, just recompute
        
    # Simpler: just recompute conv for each coordinate update
    no_improve = 0
    max_no_improve = 3000
    
    while elapsed() < DEADLINE - 20 and no_improve < max_no_improve:
        # Random coordinate
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        # Try several candidate values
        # Scale: current mean height
        mean_h = np.mean(current_h)
        candidates_val = [0.0, orig*0.5, orig*0.75, orig, orig*1.25, orig*1.5, orig*2.0, mean_h*0.1, mean_h*0.5, mean_h, mean_h*2.0, mean_h*5.0]
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if cv == orig:
                continue
            current_h[idx] = cv
            v = metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv
        
        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1
        
        # Periodic perturbation
        if no_improve % 200 == 0:
            pert = current_h.copy()
            pert += 0.1 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
    
    # Phase 4: Fine-tune with small steps
    if elapsed() < DEADLINE - 5:
        step_frac = 0.01
        for _ in range(500):
            if elapsed() > DEADLINE - 3:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
