"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft, dct, idct
from scipy.optimize import minimize

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Combines parametric initialization, random walk exploration, and DCT-domain
    optimization with a soft-max surrogate for the autocorrelation peak.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0

    def compute_value(h):
        """Compute the evaluation metric."""
        h_sum = np.sum(h)
        if h_sum == 0 or not np.all(np.isfinite(h)):
            return np.inf
        autoconv = np.convolve(h, h)
        max_conv = np.max(autoconv)
        return abs(2 * N * max_conv / (h_sum ** 2))

    def autoconv_fft(h):
        """Compute linear autoconvolution efficiently via FFT."""
        n_conv = 2 * N - 1
        n_fft = 1
        while n_fft < n_conv:
            n_fft *= 2
        H = fft(h, n=n_fft)
        conv = ifft(H * H, n=n_fft).real[:n_conv]
        return conv

    def soft_max(x, beta):
        """Numerically stable soft maximum."""
        m = np.max(x)
        if not np.isfinite(m):
            return m
        exp_vals = np.exp(beta * (x - m))
        sm = m + np.log(np.sum(exp_vals)) / beta
        return sm

    def objective_soft(h, beta=10.0):
        """
        Objective function using soft-max for the peak of the autoconvolution.
        """
        if not np.all(np.isfinite(h)):
            return 1e10
        
        h_sum = np.sum(h)
        if abs(h_sum) < 1e-10:
            return 1e10
        
        h_norm = h * (N / h_sum)
        autoconv = autoconv_fft(h_norm)
        max_conv_soft = soft_max(autoconv, beta)
        value = abs(2 * N * max_conv_soft / (N ** 2))
        return value

    # --- Phase 1: Parametric Initialization ---
    # Try simple families to get a good starting point better than ones.
    
    # Family 1: Cosine waves
    for m in range(1, 21):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.8, 0.8, 17):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = compute_value(h)
            if v < best_value:
                best_value = v
                best_h = h.copy()

    # Family 2: Smoothed block (tanh)
    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 80:
            break
        h = np.tanh(beta * (np.arange(N) - N / 2.0))
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        v = compute_value(h)
        if v < best_value:
            best_value = v
            best_h = h.copy()

    # --- Phase 2: Random Walk Exploration ---
    # Generate random walks with drift to maintain sum=N, check value.
    if time.time() < deadline - 60:
        rng = np.random.RandomState(42)
        
        for noise_scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            if time.time() > deadline - 60:
                break
            
            for trial in range(20):
                if time.time() > deadline - 60:
                    break
                
                noise = rng.randn(N) * noise_scale
                walk = np.cumsum(noise)
                
                # Subtract linear trend to make it zero-sum perturbation
                # h = 1 + walk - drift*i
                # sum(h) = N + sum(walk) - drift*sum(i) = N
                # drift = sum(walk) / sum(i)
                sum_i = np.sum(np.arange(N))
                drift = np.sum(walk) / sum_i
                
                h = 1.0 + walk - drift * np.arange(N)
                
                if not np.all(np.isfinite(h)) or abs(np.sum(h) - N) > 1e-6:
                    continue
                
                v = compute_value(h)
                if v < best_value:
                    best_value = v
                    best_h = h.copy()

    # --- Phase 3: DCT Domain Optimization ---
    # Convert best_h to DCT coefficients.
    # Optimize AC coefficients with soft-max objective.
    
    def h_to_dct(h):
        return dct(h, type=2, norm='ortho')
    
    def dct_to_h(coeffs):
        return idct(coeffs, type=2, norm='ortho')

    c_best = h_to_dct(best_h)
    c_dc = c_best[0].copy()

    # Multi-stage beta annealing in DCT domain
    betas = [1.0, 5.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 25:
            break

        def obj_dct(c_ac, beta=beta):
            c_full = np.concatenate([[c_dc], c_ac])
            h_recon = dct_to_h(c_full)

            if not np.all(np.isfinite(h_recon)):
                return 1e10

            s = np.sum(h_recon)
            if abs(s) < 1e-10:
                return 1e10

            h_norm = h_recon * (N / s)
            conv = autoconv_fft(h_norm)
            sm = soft_max(conv, beta)
            return abs(2.0 * N * sm / (N * N))

        # Try a few restarts
        rng = np.random.RandomState(42 + int(beta))
        
        for trial in range(4):
            if time.time() > deadline - 25:
                break

            if trial == 0:
                c_ac_trial = c_best[1:].copy()
            else:
                c_ac_trial = c_best[1:].copy() + 0.1 * rng.randn(N-1)

            try:
                result = minimize(
                    obj_dct,
                    c_ac_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = compute_value(h_opt)
                    if v_opt < best_value:
                        best_value = v_opt
                        best_h = h_opt.copy()
                        c_best = h_to_dct(best_h)
                        c_dc = c_best[0].copy()
            except Exception:
                continue

    # --- Phase 4: High Precision Polish ---
    # Very high beta to sharpen the max approximation.
    if time.time() < deadline - 15:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10:
                break

            def obj_polish(c_ac, beta=beta):
                c_full = np.concatenate([[c_dc], c_ac])
                h_recon = dct_to_h(c_full)

                if not np.all(np.isfinite(h_recon)):
                    return 1e10

                s = np.sum(h_recon)
                if abs(s) < 1e-10:
                    return 1e10

                h_norm = h_recon * (N / s)
                conv = autoconv_fft(h_norm)
                sm = soft_max(conv, beta)
                return abs(2.0 * N * sm / (N * N))

            try:
                result = minimize(
                    obj_polish,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )

                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc], c_ac_opt])
                h_opt = dct_to_h(c_full_opt)

                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = compute_value(h_opt)
                    if v_opt < best_value:
                        best_value = v_opt
                        best_h = h_opt.copy()
                        c_best = h_to_dct(best_h)
                        c_dc = c_best[0].copy()
            except Exception:
                continue

    return best_h
# EVOLVE-BLOCK-END
