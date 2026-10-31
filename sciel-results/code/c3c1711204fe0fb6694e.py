import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 138.0  # safety margin before 150s hard limit

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 2048  # higher resolution for better accuracy
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)

    def get_autoconv(h):
        """Return the raw autoconvolution array."""
        nfft = 2048
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
        nfft = 2048
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
    n = np.arange(N)
    center = N / 2 - 0.5

    # Uniform
    candidates.append(np.ones(N))

    # Cosine taper
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))

    # Sinc-like
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)

    # Gaussian
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))

    # Blackman window
    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))

    # Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))

    # Hanning window
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))

    # Flat top window
    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) + 
                0.277263158 * np.cos(4*np.pi*n/(N-1)) + 
                0.083578947 * np.cos(6*np.pi*n/(N-1)) + 
                0.006947368 * np.cos(8*np.pi*n/(N-1)))
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

    # Evaluate candidates
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 2: Spectral optimization - represent h as band-limited spectrum
    # The autoconvolution in time domain corresponds to |H(f)|^2 in frequency domain
    # We optimize the spectral shape to minimize the peak of the inverse transform
    
    if elapsed() < DEADLINE - 60:
        def spectral_heights(params):
            """
            params: [cutoff, w1, w2, ..., wK] where K window coefficients
            cutoff: normalized cutoff frequency (0 to 0.5)
            w: window coefficients for shaping the spectrum
            """
            K = len(params) - 1
            cutoff = params[0]
            w = params[1:]
            
            # Create frequency grid
            freqs = np.fft.rfftfreq(N)  # 0 to 0.5
            # Normalize to [0, 1]
            freqs_norm = freqs / 0.5 if freqs.max() > 0 else freqs
            
            # Ideal low-pass: 1 below cutoff, 0 above
            lp = (freqs_norm <= cutoff).astype(float)
            
            # Apply window shaping
            # Use cosine window coefficients to shape the transition
            if K > 0:
                # Smooth transition using window function
                transition_width = 0.05
                mask = np.ones_like(freqs_norm)
                for i, fi in enumerate(freqs_norm):
                    if cutoff - transition_width < fi < cutoff + transition_width:
                        # Smooth transition
                        t = (fi - (cutoff - transition_width)) / (2 * transition_width)
                        t = np.clip(t, 0, 1)
                        # Apply window
                        val = 0.5 * (1 + np.cos(np.pi * (2*t - 1)))
                        # Blend with window coefficients
                        for k in range(min(K, 5)):
                            val *= (1 + 0.1 * w[k] * np.cos(k * np.pi * t))
                        mask[i] = val
                lp *= mask
            
            # Inverse FFT to get time domain
            H_spec = lp
            h = np.fft.irfft(H_spec, n=N)
            h = np.maximum(h, 0)
            s = np.sum(h)
            if s < 1e-15:
                return np.ones(N)
            return h / s
        
        n_window_coeffs = 5
        bounds = [(0.1, 0.5)] + [(-1, 1)] * n_window_coeffs
        
        def spectral_objective(params):
            if elapsed() > DEADLINE - 10:
                return 1e10
            h = spectral_heights(params)
            return fast_metric(h)
        
        try:
            result = differential_evolution(
                spectral_objective,
                bounds,
                maxiter=30,
                popsize=8,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=True
            )
            if result.fun < 1e9:
                h_spec = spectral_heights(result.x)
                v_spec = fast_metric(h_spec)
                if v_spec < best_val:
                    best_val = v_spec
                    best_h = h_spec.copy()
        except Exception:
            pass

    # Phase 3: Gaussian Mixture optimization (keep from before, refined)
    if elapsed() < DEADLINE - 50:
        def gaussian_mixture_heights(params, n_components=6):
            h = np.zeros(N)
            x_grid = np.linspace(-0.25, 0.25, N)
            for k in range(n_components):
                amp = params[3*k]
                c_center = params[3*k+1]
                c_width = params[3*k+2]
                if c_width < 1e-6:
                    c_width = 1e-6
                h += amp * np.exp(-0.5 * ((x_grid - c_center) / c_width) ** 2)
            h = np.maximum(h, 0)
            s = np.sum(h)
            if s < 1e-15:
                return np.ones(N)
            return h / s

        n_components = 6
        n_params = 3 * n_components
        bounds = []
        for _ in range(n_components):
            bounds.append((0.0, 2.0))   # amplitude
            bounds.append((-0.25, 0.25)) # center
            bounds.append((0.01, 0.5))   # width

        def gm_objective(params):
            if elapsed() > DEADLINE - 10:
                return 1e10
            h = gaussian_mixture_heights(params, n_components)
            return fast_metric(h)

        try:
            result = differential_evolution(
                gm_objective,
                bounds,
                maxiter=40,
                popsize=10,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=True
            )
            if result.fun < 1e9:
                h_gm = gaussian_mixture_heights(result.x, n_components)
                v_gm = fast_metric(h_gm)
                if v_gm < best_val:
                    best_val = v_gm
                    best_h = h_gm.copy()
        except Exception:
            pass

    # Phase 4: Gradient-based optimization with smooth max, annealed temperatures
    def objective(x, temp=0.01):
        if elapsed() > DEADLINE - 15:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=temp)

    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)

    # Anneal from high to low temperature for better convergence
    temps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for temp in temps:
        if elapsed() > DEADLINE - 25:
            break
        try:
            result = minimize(
                lambda x: objective(x, temp), x0, method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 400}
            )
            if result.fun < 1e9:
                h_opt = np.exp(result.x - np.max(result.x))
                v = fast_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = result.x - np.max(result.x)
        except Exception:
            pass

    # Phase 5: Coordinate descent with FFT-based metric
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)

    no_improve = 0
    max_no_improve = 3000

    while elapsed() < DEADLINE - 30 and no_improve < max_no_improve:
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

    # Phase 6: Peak-targeted polish
    if elapsed() < DEADLINE - 10:
        h_curr = best_h.copy()
        val_curr = best_val
        c = get_autoconv(h_curr)
        s = np.sum(h_curr)

        top_k = 10
        peak_lags = np.argsort(c)[::-1][:top_k]

        for _ in range(300):
            if elapsed() > DEADLINE - 8:
                break

            lag = peak_lags[np.random.randint(len(peak_lags))]

            valid_pairs = []
            for i in range(N - lag):
                if h_curr[i] > 1e-6 and h_curr[i+lag] > 1e-6:
                    valid_pairs.append((i, i+lag))

            if not valid_pairs:
                continue

            n_try = min(5, len(valid_pairs))
            trial_pairs = np.random.choice(len(valid_pairs), n_try, replace=False)

            improved = False
            for ti in trial_pairs:
                i, j = valid_pairs[ti]
                for (idx, factor) in [(i, 0.95), (j, 0.95)]:
                    new_h = h_curr.copy()
                    new_h[idx] *= factor
                    if new_h[idx] < 0:
                        continue
                    v = fast_metric(new_h)
                    if v < val_curr - 1e-12:
                        h_curr = new_h
                        val_curr = v
                        improved = True
                        break
                if improved:
                    break

            if improved:
                if val_curr < best_val:
                    best_val = val_curr
                    best_h = h_curr.copy()
                if _ % 50 == 0:
                    c = get_autoconv(h_curr)
                    peak_lags = np.argsort(c)[::-1][:top_k]

    # Phase 7: Fine-tune with small paired moves
    if elapsed() < DEADLINE - 5:
        step_frac = 0.005
        for _ in range(1500):
            if elapsed() > DEADLINE - 3:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h

    # Phase 8: Final high-precision polish
    if elapsed() < DEADLINE - 2:
        h_final = best_h.copy()
        val_final = best_val
        mean_h = np.mean(h_final)

        for _ in range(300):
            if elapsed() > DEADLINE - 1:
                break
            for _ in range(3):
                idx = np.random.randint(N)
                for factor in [0.99, 0.995, 1.005, 1.01]:
                    new_h = h_final.copy()
                    new_h[idx] *= factor
                    if new_h[idx] < 0:
                        continue
                    v = fast_metric(new_h)
                    if v < val_final - 1e-14:
                        h_final = new_h
                        val_final = v
                        break

        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
