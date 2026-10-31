import numpy as np
import time

N = 600
DEADLINE = 135  # safety margin before 150s hard limit
nfft = 2048  # next power of 2 >= 2N-1 = 1199

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_conv(h):
        """Compute full autoconvolution using FFT."""
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1]
    
    def metric(h):
        """Normalized metric: max_conv / (sum_h)^2, scale-invariant."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        return np.max(fast_conv(h)) / (s * s)
    
    def softmax_max(c, temp):
        """Smooth approximation of max using log-sum-exp."""
        c = c / temp
        c = c - np.max(c)
        return temp * (np.log(np.sum(np.exp(c))))
    
    def smooth_metric(h, temp=0.01):
        """Smooth surrogate for the nonsmooth max objective."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        c = fast_conv(hn)
        return softmax_max(c, temp)
    
    # Phase 1: Evaluate structured candidates
    i = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    candidates.append(0.5 * (1 + np.cos(np.pi * i / (N - 1))))
    
    # Gaussian
    xg = (i - N/2) / (N * 0.3)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*i/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*i/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*i/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*i/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau variants
    for a, b in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3), (0.2, 0.8), (0.8, 0.2)]:
        tp = np.ones(N)
        tp[N//4:3*N//4] = a
        tp[:N//4] = b
        tp[3*N//4:] = b
        candidates.append(tp)
    
    # Sine bump
    candidates.append(np.sin(np.pi * i / (N - 1)))
    
    # Plateau in middle, zero edges
    for e in [0.1, 0.2]:
        pp = np.zeros(N)
        pp[int(e*N):N-int(e*N)] = 1.0
        candidates.append(pp)
    
    # Super-Gaussian
    for sigma in [0.15, 0.2, 0.25, 0.35]:
        xg = (i - N/2) / (N * sigma)
        candidates.append(np.exp(-0.5 * xg**4))
    
    # Triangle (peak in center)
    tri = np.ones(N)
    center = N // 2
    for idx in range(N):
        dist = abs(idx - center)
        tri[idx] = max(0.1, 1.0 - dist / (N/2))
    candidates.append(tri)
    
    best_val = 1e10
    best_h = np.ones(N)
    
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Differential evolution on Gaussian mixture parameters
    # This is the structural idea from inspiration 1 - optimize continuous parameters
    # of a Gaussian mixture to find smoother step functions
    
    DT = (1/4 - (-1/4)) / N  # interval width = 1/600
    X_GRID = -1/4 + (np.arange(N) + 0.5) * DT
    
    def gaussian_mixture_heights(params, n_components):
        """Convert parameter vector to step heights using Gaussian mixture."""
        h = np.zeros(N)
        for k in range(n_components):
            amp = params[3*k]
            center = params[3*k+1]
            width = params[3*k+2]
            if width < 1e-6:
                width = 1e-6
            h += amp * np.exp(-0.5 * ((X_GRID - center) / width) ** 2)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-15:
            return np.ones(N) / N
        return h / s
    
    n_components = 6
    n_params = 3 * n_components
    
    bounds = []
    for _ in range(n_components):
        bounds.append((0.0, 2.0))   # amplitude
        bounds.append((-0.25, 0.25)) # center
        bounds.append((0.01, 0.5))   # width
    
    def de_objective(params):
        if elapsed() > DEADLINE - 30:
            return 1e10
        h = gaussian_mixture_heights(params, n_components)
        return metric(h)
    
    try:
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=100,
            popsize=12,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True
        )
        
        h_result = gaussian_mixture_heights(result.x, n_components)
        val_result = metric(h_result)
        
        if val_result < best_val:
            best_val = val_result
            best_h = h_result.copy()
            
    except Exception:
        pass
    
    # Phase 3: Remez-like iterative equioscillation on cosine coefficients
    # (vectorized gradient computation) - refinement machinery from current program
    
    K = 12  # number of cosine harmonics
    
    def cheb_to_h(coeffs):
        """Convert cosine series coefficients to step heights (nonnegative)."""
        i = np.arange(N)
        h = np.full(N, coeffs[0])
        for k in range(1, K+1):
            h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
        return np.maximum(h, 0.0)
    
    def fit_cosine(h, K_order):
        """Fit first K_order cosine coefficients to h."""
        i = np.arange(N)
        A = np.zeros((N, K_order + 1))
        A[:, 0] = 1.0
        for k in range(1, K_order + 1):
            A[:, k] = np.cos(2 * np.pi * k * i / N)
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return coeffs
    
    coeffs = fit_cosine(best_h, K)
    
    # Precompute cosine basis matrix for fast gradient
    i_arr = np.arange(N)
    cos_basis = np.zeros((K + 1, N))
    cos_basis[0] = 1.0
    for k in range(1, K + 1):
        cos_basis[k] = np.cos(2 * np.pi * k * i_arr / N)
    
    for remez_iter in range(50):
        if elapsed() > DEADLINE - 30:
            break
        
        h = cheb_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-15:
            break
        
        c = fast_conv(h)
        max_idx = np.argmax(c)
        max_val = c[max_idx]
        
        # Vectorized gradient computation
        k = max_idx
        if k >= N:
            k = 2*N - 1 - k
        
        if k == 0:
            grad = 2.0 * cos_basis @ h / (s * s)
        else:
            n_valid = N - k
            h1 = h[:n_valid]
            h2 = h[k:k+n_valid]
            c1 = cos_basis[:, :n_valid]
            c2 = cos_basis[:, k:k+n_valid]
            grad = (c1 @ h2 + c2 @ h1) / (s * s)
        
        # Simple gradient step to reduce the peak
        grad_norm = np.max(np.abs(grad)) + 1e-15
        lr = 0.01 * max_val / grad_norm
        lr = min(lr, 0.1)
        
        new_coeffs = coeffs - lr * grad
        new_h = cheb_to_h(new_coeffs)
        
        if np.sum(new_h) > 1e-15:
            new_val = metric(new_h)
            if new_val < best_val:
                best_val = new_val
                best_h = new_h.copy()
                coeffs = new_coeffs
            else:
                new_coeffs = coeffs - lr * 0.1 * grad
                new_h = cheb_to_h(new_coeffs)
                if np.sum(new_h) > 1e-15:
                    new_val = metric(new_h)
                    if new_val < best_val:
                        best_val = new_val
                        best_h = new_h.copy()
                        coeffs = new_coeffs
    
    # Phase 4: Water-filling iteration
    current_h = best_h.copy()
    current_val = best_val
    
    for wf_iter in range(200):
        if elapsed() > DEADLINE - 25:
            break
        
        c = fast_conv(current_h)
        s = np.sum(current_h)
        if s < 1e-15:
            break
        
        max_idx = np.argmax(c)
        k_peak = max_idx
        if k_peak >= N:
            k_peak = 2*N - 1 - k_peak
        
        sensitivity = np.zeros(N)
        n_valid = N - k_peak
        sensitivity[:n_valid] += current_h[k_peak:k_peak+n_valid]
        if k_peak > 0:
            sensitivity[k_peak:] += current_h[:n_valid]
        
        sens_norm = (sensitivity - np.min(sensitivity)) / (np.max(sensitivity) - np.min(sensitivity) + 1e-10)
        
        alpha = 0.05
        adjustment = alpha * (0.5 - sens_norm) * np.mean(current_h)
        new_h = current_h + adjustment
        new_h = np.maximum(new_h, 0.0)
        
        if np.sum(new_h) < 1e-15:
            break
        
        new_val = metric(new_h)
        if new_val < current_val:
            current_val = new_val
            current_h = new_h
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            new_h2 = current_h + 0.1 * adjustment
            new_h2 = np.maximum(new_h2, 0.0)
            if np.sum(new_h2) > 1e-15:
                new_val2 = metric(new_h2)
                if new_val2 < current_val:
                    current_val = new_val2
                    current_h = new_h2
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
    
    # Phase 5: Direct coordinate descent on the full 600-dim space
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(42)
    
    no_improve = 0
    max_no_improve = 3000
    
    while elapsed() < DEADLINE - 15 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
        step_sizes = step_factors * mean_h
        candidates_val = list(orig + step_sizes) + list(orig - step_sizes) + [0.0, orig]
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-10:
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
        if no_improve % 200 == 0 and no_improve > 0:
            pert = current_h.copy()
            pert += 0.03 * np.random.randn(N) * np.mean(current_h)
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
    
    # Phase 6: Fine-tune with paired updates
    if elapsed() < DEADLINE - 5:
        step_frac = 0.003
        for _ in range(2000):
            if elapsed() > DEADLINE - 3:
                break
            i_idx, j_idx = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i_idx] += delta
            new_h[j_idx] -= delta
            if new_h[i_idx] >= 0 and new_h[j_idx] >= 0:
                v = metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    # Final verification with exact metric
    final_val = metric(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h.tolist()
