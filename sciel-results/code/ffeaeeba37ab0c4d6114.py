"""Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Uses Fourier-basis representation for efficient global search,
then full-space local refinement with cached smooth-surrogate gradients.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400
N_FFT = 1024  # next power of 2 >= 2*N-1 = 799
N_FFT_CC = 2048  # next power of 2 >= N + (2*N-1) = 1199

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    def fast_convolve(h):
        """Fast auto-convolution via FFT (pre-sized)."""
        H = np.fft.rfft(h, n=N_FFT)
        conv = np.fft.irfft(H * H, n=N_FFT)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute 2 * sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        via FFT cross-correlation.
        """
        h_rev = h[::-1]
        H_rev = np.fft.rfft(h_rev, n=N_FFT_CC)
        W = np.fft.rfft(weights, n=N_FFT_CC)
        conv_result = np.fft.irfft(H_rev * W, n=N_FFT_CC)
        return 2.0 * conv_result[N-1:2*N-1]

    def smooth_obj_grad(x, beta=80.0):
        """Compute objective and gradient of log-sum-exp smoothed max in one pass."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)
        
        exp_terms = np.exp(beta * (shifted - max_shifted))
        sum_w = np.sum(exp_terms)
        if sum_w < 1e-300:
            return 1e10, np.zeros_like(h)
        
        lse = max_c + max_shifted + np.log(sum_w) / beta
        f = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f):
            return 1e10, np.zeros_like(h)
        
        weights = exp_terms / sum_w
        grad_lse = fast_cross_corr(weights, h)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s ** 3)
        
        grad = df_dh if f >= 0 else -df_dh
        return f, grad

    rng = np.random.RandomState(42)
    i_arr = np.arange(N)
    t = np.linspace(0, 1, N, endpoint=False)

    # --- Generate a focused set of high-quality initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Low-frequency cosines (structured, smooth)
    for freq in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) - 0.5)  # shifted to allow negative sum regions

    # Bump profiles (Gaussian)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [20, 40, 60, 100, 150, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            initial_points.append(h_bump - 0.3)
            # Antisymmetric pair
            h_anti = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_anti)

    # Random Fourier synthesis (smooth, low-dim)
    for n_comp in [3, 5, 7]:
        for _ in range(4):
            freqs = rng.randint(1, 25, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Smooth random
    for _ in range(10):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(3, 15))
        initial_points.append(h_smooth)

    # Block patterns
    for block_size in [20, 50, 100]:
        for _ in range(3):
            n_blocks = N // block_size
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)

    # Sort by initial value (best first) — use fast_convolve
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)
    # Keep top 20 to focus compute
    initial_points = initial_points[:20]

    # --- Phase 1: Smooth-surrogate L-BFGS-B from each initial point ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1

        try:
            result = minimize(
                lambda x: smooth_obj_grad(x, beta=60.0)[0],
                init_h,
                jac=lambda x: smooth_obj_grad(x, beta=60.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            
            val, valid = evaluate(result.x)
            if valid and val < best_value:
                best_value = val
                best_h = result.x.copy()
        except Exception:
            continue

    # --- Phase 2: Polish with increasing beta ---
    if time.time() - start_time < DEADLINE:
        for beta in [100, 200]:
            if time.time() - start_time > DEADLINE:
                break
            
            try:
                result = minimize(
                    lambda x: smooth_obj_grad(x, beta=beta)[0],
                    best_h,
                    jac=lambda x: smooth_obj_grad(x, beta=beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                
                val, valid = evaluate(result.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = result.x.copy()
            except Exception:
                continue

    # --- Phase 3: Exact-gradient polish (subgradient of max) ---
    if time.time() - start_time < DEADLINE:
        def exact_obj_grad(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10, np.zeros_like(h)
            
            conv = fast_convolve(h)
            max_val = np.max(conv)
            idx = np.argmax(conv)
            
            raw = 2.0 * N * max_val / (s * s)
            val = abs(raw)
            
            # d(conv[idx])/d(h[i]): conv[idx] = sum_{j} h[j]*h[idx-j]
            # d/dh[i] = h[idx-i] + h[i] (when both indices valid)
            dh_conv = np.zeros(N)
            for i in range(N):
                j1 = idx - i
                j2 = i
                if 0 <= j1 < N:
                    dh_conv[i] += h[j1]
                if 0 <= j2 < N:
                    dh_conv[i] += h[j2]
            
            # d|raw|/dh = sign(raw) * d(raw)/dh
            # d(raw)/dh = 2N * (dh_conv * s - 2*max_val) / s^3
            grad_raw = 2.0 * N * (dh_conv * s - 2.0 * max_val) / (s ** 3)
            grad = grad_raw if raw >= 0 else -grad_raw
            
            return val, grad

        try:
            result = minimize(
                exact_obj_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
            )
            
            val, valid = evaluate(result.x)
            if valid and val < best_value:
                best_value = val
                best_h = result.x.copy()
        except Exception:
            pass

    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = fast_convolve(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
