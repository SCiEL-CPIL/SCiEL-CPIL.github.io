"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _evaluate(h):
    """Compute the objective value for a given height vector."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv)) / (s * s)


def _smooth_max(x, k=50.0):
    """Soft-max via log-sum-exp for smooth optimization."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(k * (x - m)))) / k


def _grad_smooth_max(x, k=50.0):
    """Gradient of the soft-max."""
    m = np.max(x)
    e = np.exp(k * (x - m))
    return e / np.sum(e)


def _project_simplex(v, s):
    """Project vector v onto the simplex {x >= 0, sum(x) = s}."""
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit
    
    # Start with the seed
    h = np.ones(N)
    best_h = h.copy()
    best_val = _evaluate(best_h)
    
    # Also try some structured initializations
    candidates = [np.ones(N)]
    
    # Symmetric patterns
    for period in [2, 4, 8, 16]:
        for phase in range(period):
            c = np.ones(N)
            c[phase::period] = -1.0
            if np.sum(c) != 0:
                candidates.append(c)
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(20):
        c = np.ones(N) * (1.0 + 0.5 * rng.randn(N))
        if np.sum(c) != 0:
            candidates.append(c)
    
    # Evaluate all candidates
    for c in candidates:
        c = c.astype(np.float64)
        val = _evaluate(c)
        if val < best_val:
            best_val = val
            best_h = c.copy()
    
    # ADMM-style optimization: minimize max(autoconv) subject to sum(h) = N
    # Using a smooth surrogate for the max
    h = best_h.copy()
    sum_target = np.sum(h)
    
    k_smooth = 30.0
    lr = 0.01
    rho = 10.0  # ADMM penalty
    
    # Initialize dual variable for sum constraint
    nu = 0.0
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Compute autoconvolution
        conv = np.convolve(h, h)
        
        # Smooth max and its gradient
        sm = _smooth_max(conv, k=k_smooth)
        grad_sm = _grad_smooth_max(conv, k=k_smooth)
        
        # Gradient of smooth max w.r.t. h: d/dh max(conv) 
        # conv[j] = sum_i h[i]*h[j-i], so d conv[j]/d h[l] = h[j-l] + h[l] (for valid indices)
        # Using FFT for efficiency
        H = np.fft.rfft(h, n=2*N)
        # Gradient of conv w.r.t. h is the circular convolution of h with ones... 
        # Actually d(conv[h,h])/dh = 2 * ifft(|H|^2) in circular sense
        # For full convolution, we need to be careful
        # Simpler: use the fact that d/dh_l [sum_j w_j * conv_j] where conv = h*h
        # = 2 * (w * h) evaluated appropriately
        
        # Let's use a direct but efficient approach via FFT
        # grad_h[l] = 2 * sum_j grad_sm[j] * (h[j-l] if j>=l else h[l-j])... this is getting complex
        # Use the identity: for full conv, grad_h = 2 * ifft(fft(grad_sm, 2N) * conj(fft(h, 2N))) truncated
        
        gs = np.zeros(2*N, dtype=np.float64)
        gs[:len(grad_sm)] = grad_sm
        hs = np.zeros(2*N, dtype=np.float64)
        hs[:N] = h
        
        GS = np.fft.rfft(gs)
        HS = np.fft.rfft(hs)
        
        grad_h_full = np.fft.irfft(GS * np.conj(HS), n=2*N)
        # The gradient of sum_j g_j * conv_j w.r.t. h is 2*(g*h) circular, but for full conv:
        # Actually it's: grad_h[l] = 2 * sum_j g_j * h[(j-l) mod ...] 
        # For full (not circular) conv, the gradient is:
        # grad_h[l] = sum_j g_j * (d conv_j / d h_l)
        # conv_j = sum_i h_i h_{j-i}
        # d conv_j / d h_l = h_{j-l} + h_l if both indices valid
        # This equals (g*h)[l] + (g*flip(h))[l] for appropriate convolutions
        
        # Simpler numerical approach: use the full conv gradient via FFT properly
        # For full convolution c = h*h (length 2N-1), 
        # dc/dh_l = h convolved with delta at position l, summed with weights
        # = 2 * (g * h)_l where * is appropriate cross-correlation
        
        # Let me just use a finite-difference-like approach via the structure
        # Actually, the gradient of f(h) = max_j (h*h)_j w.r.t. h is:
        # grad_h = 2 * P * g where P is the appropriate projection
        # For simplicity, let's use: grad_h[l] = 2 * sum_j g_j * h[j-l] (for 0<=j-l<N)
        
        # This is just 2 * (g_fullconv h)[l] where g is padded
        # Use scipy or manual
        grad_h = np.zeros(N)
        for l in range(N):
            # conv_j contains h[j-l] when j>=l and h[l-j]... no
            # conv_j = sum_{i=0}^{N-1} h_i * h_{j-i} where h_{j-i}=0 if j-i not in [0,N-1]
            # d conv_j / d h_l: if j-l in [0,N-1], contributes h_{j-l}; if l in [0,N-1] (always), contributes h_l
            # Wait, conv_j = sum_i h_i h_{j-i}, so d/dh_l = sum_i [delta_{i,l} h_{j-i} + h_i delta_{j-i,l}]
            # = h_{j-l} (if 0<=j-l<N) + h_l (if 0<=j-l<N)... no
            # = h_{j-l} if j-l in [0,N-1] plus h_l if j-l in [0,N-1]... 
            # Actually: d/dh_l sum_i h_i h_{j-i} = h_{j-l} + h_{j-l} if j-l in range? No.
            # = sum_i (delta_{i,l} * h_{j-i} + h_i * delta_{i,j-l})
            # = h_{j-l} (from first term, i=l) + h_l (from second term, i=j-l) if j-l in [0,N-1]
            # So d conv_j / d h_l = h_{j-l} + h_{j-l} = 2*h_{j-l} if 0<=j-l<N, else 0? 
            # No! First term: i=l, gives h_{j-l} (valid if 0<=j-l<N)
            # Second term: i=j-l, gives h_{j-l}... wait no, gives h_i = h_{j-l}
            # Hmm, both give h_{j-l}. So it's 2*h_{j-l} if 0<=j-l<N.
            # But that's only when j-l is a valid index.
            
            pass  # This is O(N^2), too slow for N=400 in Python loop
        
        # Better: use the FFT-based approach properly
        # The gradient of the full autoconvolution energy with weights g is:
        # grad_h = 2 * ifft(fft(g, 2N) * conj(fft(h, 2N))) [0:N] -- this gives circular
        # For full conv, we need to zero-pad properly
        
        gs_pad = np.zeros(2*N)
        gs_pad[:len(grad_sm)] = grad_sm
        hs_pad = np.zeros(2*N)
        hs_pad[:N] = h
        
        # The full convolution gradient: (g * h) where * is full convolution, but we want
        # sum_j g_j * d(conv_j)/dh_l = sum_j g_j * 2*h_{j-l} (for valid j-l)
        # This is 2 * (g_full * h)[l] where g_full is the full convolution of g with h
        # (g_full * h)[l] = sum_j g_j * h[j-l] (with h[k]=0 for k outside [0,N-1])
        # This is just the cross-correlation, computable via FFT
        
        g_pad = np.zeros(2*N - 1)
        g_pad[:len(grad_sm)] = grad_sm
        h_pad = np.zeros(2*N - 1)
        h_pad[:N] = h
        
        # Cross-correlation: sum_j g_j h[j-l] = ifft(fft(g) * conj(fft(h)))
        G = np.fft.fft(g_pad, n=2*N)
        H = np.fft.fft(h_pad, n=2*N)
        cross_corr = np.fft.ifft(G * np.conj(H)).real
        grad_h = 2.0 * cross_corr[:N]
        
        # Constraint: sum(h) = sum_target
        # ADMM update: h_new = argmin 0.5*||h||^2 + <grad, h> + rho/2 * (sum(h) - sum_target + nu/rho)^2
        # = argmin over h: 0.5*||h||^2 + <grad, h> + rho/2*(sum(h) - c)^2 where c = sum_target - nu/rho
        # This is a quadratic in sum(h) only if grad is constant... it's not.
        # Actually with just the sum constraint, the minimizer is:
        # h_new = h - lr * grad_h + (1/rho) * (sum_target - sum(h) + nu) * ones
        # Let's use a simple projected gradient with the sum constraint
        
        # Update nu
        nu += sum_target - np.sum(h)
        
        # Gradient step
        h_new = h - lr * grad_h
        
        # Project onto {sum(h) = sum_target}
        h_new = h_new + (sum_target - np.sum(h_new)) / N
        
        # Check improvement
        val = _evaluate(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if it % 100 == 99:
            if best_val > 2.0:
                lr *= 0.5
            k_smooth = max(5.0, k_smooth * 0.95)
    
    # Final polish: try local search around best_h
    t1 = time.time()
    if t1 < deadline - 20:
        for trial in range(50):
            if time.time() > deadline:
                break
            perturbation = 0.01 * np.random.randn(N)
            h_trial = best_h + perturbation
            # Ensure sum is reasonable
            h_trial = h_trial + (sum_target - np.sum(h_trial)) / N
            val = _evaluate(h_trial)
            if val < best_val:
                best_val = val
                best_h = h_trial.copy()
    
    # Final check
    final_val = _evaluate(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
# EVOLVE-BLOCK-END
