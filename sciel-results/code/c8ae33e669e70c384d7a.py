import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024  # next power of 2 >= 2*N-1 = 1199
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    def get_autoconv(h):
        """Return the raw autoconvolution array."""
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2*N-1]
    
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
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)
    
    # Evaluate uniform baseline
    h_uniform = np.ones(N)
    best_val = fast_metric(h_uniform)
    best_h = h_uniform.copy()
    
    # Phase 1: Try structured candidates
    candidates = []
    i = np.arange(N)
    center = N / 2 - 0.5
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Cosine taper
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    # Sinc-like
    width = 0.3
    sinc_h = np.sinc((i - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)
    
    # Gaussian
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))
    
    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    candidates.append(np.clip(blackman, 0, None))
    
    # Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hamming, 0, None))
    
    # Hanning window
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * i / (N - 1))
    candidates.append(np.clip(hanning, 0, None))
    
    # Flat top window
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*i/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*i/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*i/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*i/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    
    # Two-plateau: high at edges, low in center
    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)
    
    # Inverse: low at edges, high in center
    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)
    
    # Triangular (Bartlett)
    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)
    
    # Asymmetric candidates (key insight from inspiration)
    # Skewed Gaussian
    for skew in [0.3, -0.3, 0.5, -0.5]:
        xg = (i - center - skew * N * 0.1) / (N * 0.3)
        asym_gauss = np.exp(-0.5 * xg**2)
        candidates.append(asym_gauss)
    
    # Skewed cosine
    for shift in [0.1, -0.1, 0.2, -0.2]:
        x_shifted = np.linspace(0, np.pi * (1 + shift), N)
        asym_cos = 0.5 * (1 + np.cos(x_shifted))
        candidates.append(asym_cos)
    
    # Step functions with different breakpoints
    for bp in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
        step_h = np.ones(N)
        step_h[int(N*bp):] = 0.3
        candidates.append(step_h)
        step_h2 = np.ones(N)
        step_h2[:int(N*bp)] = 0.3
        candidates.append(step_h2)
    
    # Evaluate candidates
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Gradient-based optimization with smooth max, multiple temperatures
    # Use projected gradient descent from multiple asymmetric initializations
    def objective(x, temp=0.01):
        if elapsed() > DEADLINE - 15:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=temp)
    
    # Create multiple asymmetric starting points
    np.random.seed(42)
    start_points = []
    
    # Log of best candidate
    x0_best = np.log(best_h + 1e-15)
    x0_best = x0_best - np.max(x0_best)
    start_points.append(x0_best)
    
    # Random asymmetric perturbations
    for _ in range(5):
        x_rand = x0_best.copy()
        # Add asymmetric perturbation
        pert = np.random.randn(N) * 0.1
        # Make it asymmetric by adding a linear trend
        pert += np.linspace(-0.2, 0.2, N) * np.random.uniform(0.5, 1.5)
        x_rand += pert
        x_rand = x_rand - np.max(x_rand)
        start_points.append(x_rand)
    
    # Try each starting point with annealed temperatures
    for x0 in start_points:
        if elapsed() > DEADLINE - 30:
            break
        
        for temp in [0.02, 0.01, 0.005, 0.002]:
            if elapsed() > DEADLINE - 20:
                break
            try:
                result = minimize(
                    lambda x: objective(x, temp), x0, method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 300}
                )
                if result.success or result.fun < 1e9:
                    h_opt = np.exp(result.x - np.max(result.x))
                    v = fast_metric(h_opt)
                    if v < best_val:
                        best_val = v
                        best_h = h_opt.copy()
                        x0 = result.x - np.max(result.x)
            except Exception:
                pass
    
    # Phase 3: Simulated annealing fine-tune (from inspiration)
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    T_start = 0.05
    T_end = 0.001
    n_sa_iters = 2000
    
    for sa_iter in range(n_sa_iters):
        if elapsed() > DEADLINE - 25:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (sa_iter / n_sa_iters)
        
        # Generate perturbation
        perturbation_scale = T * np.mean(current_h)
        new_h = current_h.copy()
        idx = np.random.randint(N)
        delta = np.random.randn() * perturbation_scale
        new_h[idx] = max(0.0, new_h[idx] + delta)
        
        new_val = fast_metric(new_h)
        delta_val = new_val - current_val
        
        # Accept if better or with probability exp(-delta_val / T)
        if delta_val < 0 or np.random.rand() < np.exp(-delta_val / max(T, 1e-10)):
            current_h = new_h
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
    
    # Phase 4: Coordinate descent with FFT-based metric
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    no_improve = 0
    max_no_improve = 5000
    
    while elapsed() < DEADLINE - 25 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        mean_h = np.mean(current_h)
        step_factors = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0, orig]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-12:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
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
        
        # Periodic perturbation to escape local minima
        if no_improve % 200 == 0:
            pert = current_h.copy()
            pert += 0.08 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
    
    # Phase 5: Fine-tune with small paired moves
    if elapsed() < DEADLINE - 5:
        step_frac = 0.005
        for _ in range(1000):
            if elapsed() > DEADLINE - 3:
                break
            i_idx, j_idx = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i_idx] += delta
            new_h[j_idx] -= delta
            if new_h[i_idx] >= 0 and new_h[j_idx] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
