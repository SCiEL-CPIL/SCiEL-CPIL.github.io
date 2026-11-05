import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val if valid else np.inf

    val_seed = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    fft_size_cc = 1
    while fft_size_cc < N + 2 * N - 1:
        fft_size_cc *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        using FFT-based cross-correlation.
        """
        h_rev = h[::-1]
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        
        # Extract the relevant part
        grad_lse = 2.0 * conv_result[N-1:2*N-1]
        
        return grad_lse

    def smooth_objective_and_grad(x, beta=150.0):
        """Compute both objective and gradient in one pass to avoid redundant FFTs."""
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
        sum_weights = np.sum(exp_terms)
        if sum_weights < 1e-300:
            return 1e10, np.zeros_like(h)
        
        lse = max_c + max_shifted + np.log(sum_weights) / beta
        f_val = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f_val):
            return 1e10, np.zeros_like(h)
        
        # Gradient of lse w.r.t. h
        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)
        
        # Gradient of f = 2*N*lse/s^2
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        # Gradient of |f|
        if f_val >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return f_val, grad

    def smooth_objective(x, beta=150.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=150.0):
        """Analytic gradient of the smooth surrogate."""
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    rng = np.random.RandomState(42)

    # --- Generate focused initial points emphasizing spectral-null designs ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    t = np.linspace(0, 1, N, endpoint=False)
    i_arr = np.arange(N)
    ones = np.ones(N)
    ones_norm_sq = np.dot(ones, ones)

    # --- High-priority: spectral null designs (target DC null to reduce peak autocorr) ---
    # Random vectors with DC projected out, small DC offset
    for _ in range(20):
        v = rng.normal(0, 1, N)
        dc_component = np.dot(v, ones) / ones_norm_sq * ones
        v_zero_dc = v - dc_component
        dc_offset = rng.uniform(0.05, 0.3) * np.linalg.norm(v_zero_dc) / N
        h_spectral = v_zero_dc + dc_offset * ones
        initial_points.append(h_spectral)
    
    # Fourier-domain spectral nulls with structured phase patterns
    for _ in range(15):
        spectrum = rng.normal(0, 1, N // 2 + 1) + 1j * rng.normal(0, 1, N // 2 + 1)
        spectrum[0] = 0.0
        spectrum_full = np.zeros(N, dtype=complex)
        spectrum_full[:N // 2 + 1] = spectrum
        spectrum_full[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
        h_fourier_null = np.real(np.fft.ifft(spectrum_full))
        dc_offset = rng.uniform(0.05, 0.3) * np.linalg.norm(h_fourier_null) / N
        h_fourier_null += dc_offset
        initial_points.append(h_fourier_null)
    
    # Anti-periodic patterns: h[i] = -h[i + N//2] reduces even-lag autocorrelation
    for _ in range(8):
        half = N // 2
        h_first = rng.normal(0, 1, half)
        h_anti = np.concatenate([h_first, -h_first])
        # Add small DC
        dc_offset = rng.uniform(0.02, 0.1) * np.linalg.norm(h_anti) / N
        h_anti += dc_offset
        initial_points.append(h_anti)
    
    # Smoothed anti-periodic
    for _ in range(5):
        half = N // 2
        h_first = gaussian_filter1d(rng.normal(0, 1, half), sigma=rng.uniform(3, 10))
        h_anti = np.concatenate([h_first, -h_first])
        dc_offset = rng.uniform(0.02, 0.1) * np.linalg.norm(h_anti) / N
        h_anti += dc_offset
        initial_points.append(h_anti)

    # Low-frequency cosines (these naturally have structure that can reduce peak)
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        # Anti-phase version
        initial_points.append(-np.cos(2 * np.pi * freq * t))

    # Bimodal sparse patterns
    for _ in range(8):
        h = np.zeros(N)
        k = rng.randint(80, 250)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        # Add small DC
        dc_offset = rng.uniform(0.02, 0.1) * np.linalg.norm(h) / N
        h += dc_offset
        initial_points.append(h)

    # Smooth random
    for _ in range(8):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(3, 15))
        # Normalize and add small DC if needed
        s = np.sum(h_smooth)
        if abs(s) < 0.1 * np.linalg.norm(h_smooth):
            h_smooth += rng.uniform(0.05, 0.2) * np.linalg.norm(h_smooth) / N
        initial_points.append(h_smooth)

    # Block patterns with alternating signs
    for block_size in [30, 50, 80]:
        n_blocks = N // block_size
        for _ in range(3):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = (-1) ** b if rng.random() > 0.3 else rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            dc_offset = rng.uniform(0.02, 0.1) * np.linalg.norm(h_blocks) / N
            h_blocks += dc_offset
            initial_points.append(h_blocks)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)
    
    # Keep only top 40 to free up time for optimization
    initial_points = initial_points[:40]

    # --- Two-stage optimization: coarse then fine ---
    # Stage 1: Coarse optimization with lower beta from top initial points
    for h0 in initial_points:
        if time.time() - start_time > DEADLINE - 30:
            break
        try:
            res = minimize(
                lambda x: smooth_objective(x, beta=80.0),
                h0,
                jac=lambda x: smooth_grad(x, beta=80.0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            h_opt = res.x
            val_opt = evaluate(h_opt)
            if val_opt < best_value:
                best_value = val_opt
                best_h = h_opt
        except Exception:
            pass

    # Stage 2: Fine optimization with higher beta on best result
    if time.time() - start_time < DEADLINE - 20:
        try:
            res = minimize(
                lambda x: smooth_objective(x, beta=200.0),
                best_h,
                jac=lambda x: smooth_grad(x, beta=200.0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = res.x
            val_opt = evaluate(h_opt)
            if val_opt < best_value:
                best_value = val_opt
                best_h = h_opt
        except Exception:
            pass

    # Stage 3: Polish with very high beta
    if time.time() - start_time < DEADLINE - 10:
        try:
            res_polish = minimize(
                lambda x: smooth_objective(x, beta=400.0),
                best_h,
                jac=lambda x: smooth_grad(x, beta=400.0),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_polished = res_polish.x
            val_polished = evaluate(h_polished)
            if val_polished < best_value:
                best_value = val_polished
                best_h = h_polished
        except Exception:
            pass

    # --- Gradient-based local search with decreasing step sizes ---
    if time.time() - start_time < DEADLINE - 5:
        for step_scale in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if time.time() - start_time > DEADLINE:
                break
            for _ in range(10):
                if time.time() - start_time > DEADLINE:
                    break
                # Get gradient direction
                _, grad = smooth_objective_and_grad(best_h, beta=400.0)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-15:
                    break
                # Step in negative gradient direction with noise
                noise = rng.normal(0, step_scale * 0.1, N)
                step = -grad / grad_norm * step_scale * np.linalg.norm(best_h) + noise
                h_candidate = best_h + step
                try:
                    val_candidate = evaluate(h_candidate)
                    if val_candidate < best_value:
                        best_value = val_candidate
                        best_h = h_candidate
                except Exception:
                    pass

    return best_h
