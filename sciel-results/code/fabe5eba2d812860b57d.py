import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Structural approach: Parameterize h by coefficients of a low-order rational
    generating function. This enforces smooth, structured autocorrelation decay
    and reduces the optimization to a few parameters.
    
    The overlap at shift k is: C_k = sum_i h_i (1 - h_{i-k})
    For h_i = 0.5 + d_i, C_k = N/4 - sum_i d_i d_{i+k}
    Minimizing max C_k means maximizing min autocorr_k, but since autocorr_0 > 0,
    some lags will have C_k > N/4. The uniform h=0.5 gives C_k = N/4 for all k.
    
    We use a rational generating function to create smooth deviations that
    might achieve a better peak correlation profile than random or sinusoidal
    candidates.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def get_max_overlap(h):
        """Compute max overlap of h with shifted (1-h) for all non-zero shifts."""
        # C_k = sum_i h_i * (1 - h_{(i-k) % N}) for k = 1, ..., N-1
        # Using FFT for speed
        h_fft = np.fft.fft(h)
        # Autocorrelation of h
        acf = np.fft.ifft(np.abs(h_fft)**2).real
        # C_k = N/2 - acf_k for k != 0 (since sum h = N/2)
        # But we need circular correlation
        # Actually, let's compute directly for correctness
        max_ov = 0.0
        for k in range(1, N):
            shifted = np.roll(h, k)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    def project_to_feasible(h):
        """Project h onto feasible set: 0 <= h_i <= 1, sum(h) = N/2."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(100):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-14:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        # Final exact correction
        residual = N / 2.0 - np.sum(h)
        if abs(residual) > 1e-15:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h
    
    def rational_sequence(coeffs, order):
        """
        Generate a sequence from a rational generating function.
        h[n] = sum_{j=1}^{order} coeffs[j] * r_j^n where r_j are roots.
        
        Actually, let's use a simpler parameterization:
        h[n] = 0.5 + A * cos(2*pi*freq*n + phase) + B * sin(2*pi*freq2*n + phase2)
        But to be more general, use a sum of a few sinusoids with optimized
        frequencies, amplitudes, and phases.
        
        Better: use a low-order AR(2) process parameterized by coefficients.
        h[n] = 0.5 + x[n] where x[n] = a*x[n-1] + b*x[n-2] + noise
        But we want deterministic. Let's use a Fourier series with a few terms.
        """
        # Parameterization: h[n] = 0.5 + sum_{k=1}^{K} (a_k cos(2*pi*k*n/N + phi_k) + b_k sin(2*pi*k*n/N + psi_k))
        # This is equivalent to a truncated Fourier series.
        # coeffs = [a_1, phi_1, b_1, psi_1, a_2, phi_2, b_2, psi_2, ...]
        # For simplicity, use K=3 harmonics with amplitude and phase only.
        # coeffs = [a_1, phi_1, a_2, phi_2, a_3, phi_3]
        # h[n] = 0.5 + a_1*cos(2*pi*1*n/N + phi_1) + a_2*cos(2*pi*2*n/N + phi_2) + a_3*cos(2*pi*3*n/N + phi_3)
        
        n = np.arange(N)
        h = np.full(N, 0.5)
        
        # Use up to 5 harmonics
        max_harmonics = 5
        for k in range(1, max_harmonics + 1):
            if 2*(k-1) < len(coeffs):
                a_k = coeffs[2*(k-1)]
                phi_k = coeffs[2*(k-1)+1]
                h += a_k * np.cos(2 * np.pi * k * n / N + phi_k)
        
        return project_to_feasible(h)
    
    def objective(params):
        """Optimization objective: max overlap using LogSumExp smoothing."""
        h = rational_sequence(params, 5)
        # Compute all overlaps
        overlaps = []
        for k in range(1, N):
            shifted = np.roll(h, k)
            ov = np.sum(h * (1.0 - shifted))
            overlaps.append(ov)
        overlaps = np.array(overlaps)
        
        # LogSumExp for smooth max
        beta = 50.0
        max_ov = np.max(overlaps)
        if max_ov > 500:
            max_ov = 500.0
        lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
        return lse
    
    # Start with uniform
    best_h = np.full(N, 0.5)
    best_score = get_max_overlap(best_h)  # Should be 25.0
    
    # Multi-restart optimization over Fourier coefficients
    rng = np.random.default_rng(42)
    
    max_params = 10  # 5 harmonics x 2 params each
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize parameters
        if restart == 0:
            params0 = np.zeros(max_params)
        else:
            # Random initialization with small amplitudes
            params0 = np.zeros(max_params)
            for k in range(1, 6):
                params0[2*(k-1)] = rng.uniform(-0.3, 0.3)
                params0[2*(k-1)+1] = rng.uniform(0, 2*np.pi)
        
        bounds = []
        for k in range(5):
            bounds.append((-0.5, 0.5))  # Amplitude
            bounds.append((0, 2*np.pi))  # Phase
        
        try:
            res = scipy.optimize.minimize(
                objective,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = rational_sequence(res.x, 5)
            score = get_max_overlap(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try some specific structured candidates
    candidates = []
    
    # Sinusoidal candidates
    for freq in range(1, 15):
        for phase in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            n = np.arange(N)
            h = 0.5 + 0.3 * np.cos(2 * np.pi * freq * n / N + phase)
            h = project_to_feasible(h)
            candidates.append(h)
    
    # Binary-like candidates with smoothing
    for _ in range(100):
        h = np.zeros(N)
        h[:N//2] = 1.0
        idx = np.random.permutation(N)
        h = h[idx]
        # Smooth it
        kernel = np.array([0.2, 0.3, 0.2, 0.3, 0.2])
        kernel = kernel / np.sum(kernel)
        h_smooth = np.convolve(h, kernel, mode='same')
        h_smooth = project_to_feasible(h_smooth)
        candidates.append(h_smooth)
    
    # Evaluate candidates
    for h in candidates:
        if time.time() > deadline - 2:
            break
        score = get_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final verification
    total = np.sum(best_h)
    assert abs(total - N/2) < 1e-8, f"Sum constraint violated: {total}"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1 + 1e-10), "h out of [0,1]"
    
    # Clean up numerical noise
    best_h = np.clip(best_h, 0.0, 1.0)
    # Ensure exact sum
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
