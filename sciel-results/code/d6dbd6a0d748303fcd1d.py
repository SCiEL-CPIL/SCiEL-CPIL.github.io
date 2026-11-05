"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses spectral domain optimization with a soft-max surrogate for the autocorrelation peak.
    Optimizing the log-magnitude of the FFT coefficients helps flatten the power spectrum,
    which minimizes the peak of the autocorrelation.
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
        This is differentiable and allows L-BFGS-B to work effectively.
        """
        if not np.all(np.isfinite(h)):
            return 1e10
        
        h_sum = np.sum(h)
        if abs(h_sum) < 1e-10:
            return 1e10
        
        h_norm = h * (N / h_sum)
        autoconv = autoconv_fft(h_norm)
        # We want to minimize the max positive value of the autoconvolution
        # Note: np.max on signed entries. If max is negative, value is abs(2N*neg/sum^2)
        # But typically for sum(h)>0, the peak at lag 0 is positive.
        # The problem statement says "max is taken over its signed entries".
        # So if the max entry is negative, the value is negative? No, abs() is applied outside.
        # value = abs(2 * N * np.max(autoconv) / sum^2)
        # If max(autoconv) is negative, value is positive (due to abs).
        # However, for any real signal, autoconv[0] = sum(h^2) >= 0.
        # So the max is always >= 0.
        
        max_conv_soft = soft_max(autoconv, beta)
        value = abs(2 * N * max_conv_soft / (N ** 2))
        return value

    # Strategy: 
    # 1. Optimize in the spectral domain (log-magnitude of FFT) to flatten spectrum.
    # 2. Refine in the time domain using the soft-max objective.

    np.random.seed(42)

    # --- Phase 1: Spectral Domain Optimization ---
    # We represent h by its FFT coefficients.
    # To ensure h is real, we enforce conjugate symmetry.
    # We optimize the log-magnitudes and phases.
    # However, for autocorrelation, only the magnitude spectrum matters (Wiener-Khinchin).
    # Minimize the variance of the power spectrum |H(f)|^2.
    
    def objective_spectral(params):
        """
        params: [log_mag_0, ..., log_mag_{N/2}, phase_1, ..., phase_{N/2-1}]
        Note: DC (bin 0) is real and non-negative. 
        Bin N/2 (if N even) is real.
        Other bins come in conjugate pairs.
        """
        if not np.all(np.isfinite(params)):
            return 1e10
            
        # Split params
        # N=400 is even.
        # Bins: 0, 1, ..., 199, 200, 201, ..., 399
        # Bin 0: DC (Real)
        # Bin 200: Nyquist (Real)
        # Bins 1..199: Complex, conjugate symmetric with 201..399
        
        # We will optimize:
        # log_mag[0] (DC)
        # log_mag[1..199]
        # log_mag[200] (Nyquist)
        # phase[1..199]
        
        # Total params: 1 + 199 + 1 + 199 = 400
        
        log_mag_dc = params[0]
        log_mag_ac = params[1:200] # 1..199
        log_mag_nyq = params[200]
        phases = params[201:400] # 199 phases
        
        # Construct H
        H = np.zeros(N, dtype=complex)
        
        # DC
        H[0] = np.exp(log_mag_dc)
        
        # Nyquist
        H[N//2] = np.exp(log_mag_nyq)
        
        # AC bins
        for k in range(1, N//2):
            mag = np.exp(log_mag_ac[k-1])
            phase = phases[k-1]
            H[k] = mag * np.exp(1j * phase)
            H[N-k] = np.conj(H[k])
            
        # Inverse FFT to get h
        h = ifft(H).real
        
        if not np.all(np.isfinite(h)):
            return 1e10
            
        h_sum = np.sum(h)
        if abs(h_sum) < 1e-10:
            return 1e10
            
        # Normalize to sum = N
        h_norm = h * (N / h_sum)
        
        # Objective: Minimize peak of autoconvolution
        # Use soft max for differentiability
        autoconv = autoconv_fft(h_norm)
        max_conv_soft = soft_max(autoconv, beta=10.0)
        
        # Add a penalty for non-flat spectrum? 
        # Actually, minimizing the peak of autoconv IS minimizing the spectral variance.
        # But we can add a small regularization to keep magnitudes close to 1 to avoid scaling issues?
        # The normalization handles the scale.
        
        return abs(2 * N * max_conv_soft / (N ** 2))

    # Initial guess for spectral parameters
    # Start with flat spectrum (all mags = 1, random phases)
    # log(1) = 0
    init_params = np.zeros(N)
    # Random phases
    init_params[201:400] = np.random.uniform(-np.pi, np.pi, size=199)
    
    # Run spectral optimization
    try:
        result_spec = minimize(
            objective_spectral,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result_spec.fun < best_value:
            # Reconstruct h
            params_opt = result_spec.x
            H_opt = np.zeros(N, dtype=complex)
            H_opt[0] = np.exp(params_opt[0])
            H_opt[N//2] = np.exp(params_opt[200])
            for k in range(1, N//2):
                mag = np.exp(params_opt[1 + k - 1])
                phase = params_opt[201 + k - 1]
                H_opt[k] = mag * np.exp(1j * phase)
                H_opt[N-k] = np.conj(H_opt[k])
                
            h_spec = ifft(H_opt).real
            h_sum_spec = np.sum(h_spec)
            if abs(h_sum_spec) > 1e-10:
                h_spec = h_spec * (N / h_sum_spec)
                val_spec = compute_value(h_spec)
                if val_spec < best_value:
                    best_value = val_spec
                    best_h = h_spec.copy()
    except Exception:
        pass

    # --- Phase 2: Time Domain Refinement ---
    # Use the best h found so far as initialization for time-domain optimization
    # with the soft-max objective.
    
    if time.time() < deadline - 10.0:
        # Try a few random restarts in time domain, initialized near best_h or random
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            
            if trial == 0:
                h_init = best_h.copy()
            else:
                h_ac = np.random.randn(N)
                h_ac -= np.mean(h_ac)
                h_init = h_ac + 1.0
                h_init = h_init * (N / np.sum(h_init))
            
            # Increase beta for sharper max approximation during refinement
            # But start with moderate beta for smoothness
            beta_start = 5.0
            beta_end = 50.0
            
            # Simple annealing of beta
            for step in range(5):
                beta = beta_start + (beta_end - beta_start) * (step / 4.0)
                
                def obj_curr(h):
                    return objective_soft(h, beta=beta)
                
                try:
                    res = minimize(
                        obj_curr,
                        h_init,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    h_opt = res.x
                    h_sum_opt = np.sum(h_opt)
                    if abs(h_sum_opt) > 1e-10:
                        h_opt = h_opt * (N / h_sum_opt)
                        val_opt = compute_value(h_opt)
                        if val_opt < best_value:
                            best_value = val_opt
                            best_h = h_opt.copy()
                            h_init = h_opt.copy()
                except Exception:
                    break

    return best_h
# EVOLVE-BLOCK-END
