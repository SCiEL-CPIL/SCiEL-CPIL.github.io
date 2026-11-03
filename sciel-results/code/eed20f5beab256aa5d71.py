"""Structural shift: Chebyshev spectral envelope optimization via Remez-like exchange + phase retrieval."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Compute grad_x via FFT correlation
    h = np.zeros(2 * N - 1)
    h[N-1 : 2*N-1] = grad_ac[0 : N]
    h[0 : N-1] = grad_ac[2*N-2 : N-1 : -1]
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    X_f = np.fft.rfft(x, n=L_corr)
    H_f = np.fft.rfft(h, n=L_corr)
    
    corr1 = np.fft.irfft(np.conj(H_f) * X_f, n=L_corr)[:N]
    
    h_rev = h[::-1]
    H_rev_f = np.fft.rfft(h_rev, n=L_corr)
    corr2 = np.fft.irfft(np.conj(H_rev_f) * X_f, n=L_corr)[:N]
    
    grad_x = corr1 + corr2
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_envelope_to_time(amp, phase, N_time=N):
    """Convert spectral envelope (amplitude + phase) to time domain."""
    L_fft = 1
    while L_fft < 2 * N_time:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    # Interpolate amp and phase to n_freq if needed
    if len(amp) != n_freq:
        freqs_orig = np.linspace(0, 0.5, len(amp))
        freqs_new = np.linspace(0, 0.5, n_freq)
        amp_interp = np.interp(freqs_new, freqs_orig, amp)
        phase_interp = np.interp(freqs_new, freqs_orig, phase)
    else:
        amp_interp = amp
        phase_interp = phase
    
    spectrum = amp_interp * np.exp(1j * phase_interp)
    spectrum[0] = amp_interp[0]  # Keep DC real and positive
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N_time]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _remez_envelope_optimize(freqs, amp_init, num_cheb=20, iters=100, tol=1e-8):
    """
    Remez-like exchange algorithm to optimize Chebyshev coefficients
    of the spectral envelope to minimize L_inf norm of periodic autoconvolution.
    
    The periodic autoconvolution is the inverse DFT of amp^2.
    We minimize max_k |IDFT(amp^2)[k]| by adjusting amp via Chebyshev coefficients.
    """
    n_freq = len(freqs)
    
    # Map frequencies to [-1, 1] for Chebyshev
    t = 2 * freqs - 1
    t = np.clip(t, -1, 1)
    
    # Initialize Chebyshev coefficients from amp_init
    # Least squares fit: amp = sum_k c_k * T_k(t)
    cheb_matrix = np.zeros((n_freq, num_cheb))
    for k in range(num_cheb):
        angles = k * np.arccos(t)
        cheb_matrix[:, k] = np.cos(angles)
    
    coeffs, _, _, _ = np.linalg.lstsq(cheb_matrix, amp_init, rcond=None)
    
    best_coeffs = coeffs.copy()
    best_val = np.inf
    best_amp = amp_init.copy()
    
    # Exchange set: indices where |IDFT(amp^2)| is large
    exchange_set = np.linspace(0, n_freq - 1, num_cheb + 1, dtype=int)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        # Evaluate current amp
        amp = np.zeros(n_freq)
        for k in range(num_cheb):
            angles = k * np.arccos(t)
            amp += coeffs[k] * np.cos(angles)
        amp = np.maximum(0.01, np.abs(amp))
        
        # Compute periodic autoconvolution: IDFT(amp^2)
        ac_periodic = np.fft.ifft(amp ** 2).real
        
        # Find the exchange set: indices with largest |ac_periodic|
        abs_ac = np.abs(ac_periodic)
        new_exchange_set = np.argsort(abs_ac)[::-1][:num_cheb + 1]
        
        # Check convergence
        max_ac = np.max(abs_ac)
        if it > 10 and abs(max_ac - best_val) < tol:
            break
        best_val = max_ac
        
        # Solve minimax problem on exchange set
        # We want to minimize max |ac_periodic[k]| for k in exchange_set
        # ac_periodic[k] = (1/n_freq) * sum_j amp[j]^2 * exp(-2*pi*i*j*k/n_freq)
        # This is complex; use a simpler approach:
        # Adjust coefficients to reduce the peak values
        
        # Compute gradient of max_ac w.r.t. coeffs
        # d(amp[j])/d(coeffs[k]) = T_k(t[j])
        # d(ac_periodic[m])/d(coeffs[k]) = (2/n_freq) * sum_j amp[j] * T_k(t[j]) * exp(-2*pi*i*j*m/n_freq)
        
        # For simplicity, use a steepest descent approach on the exchange set
        # with a small step size
        
        # Compute current ac on exchange set
        ac_ex = ac_periodic[new_exchange_set]
        abs_ac_ex = np.abs(ac_ex)
        
        # Find the worst point
        worst_idx = np.argmax(abs_ac_ex)
        worst_global_idx = new_exchange_set[worst_idx]
        
        # Compute gradient direction: reduce |ac[worst_global_idx]|
        # ac[m] = (1/n) * sum_j amp[j]^2 * exp(-2*pi*i*j*m/n)
        # d|ac[m]|/d(coeffs[k]) = Re(conj(ac[m]) * d(ac[m])/d(coeffs[k]))
        
        # Simplified: just do a small step in the direction that reduces the peak
        # Use the sign of the real part of the contribution
        
        step = 0.01 / (1 + it * 0.01)
        
        # Compute the effect of each coefficient on the worst point
        for k in range(num_cheb):
            # Contribution of coefficient k to ac[worst_global_idx]
            # d(ac[m])/d(c_k) = (2/n) * sum_j amp[j] * T_k(t[j]) * exp(-2*pi*i*j*m/n)
            phase_factor = np.exp(-2j * np.pi * worst_global_idx * np.arange(n_freq) / n_freq)
            d_ac_dc = (2.0 / n_freq) * np.sum(amp * cheb_matrix[:, k] * phase_factor)
            
            # Update coefficient to reduce |ac|
            if abs(ac_periodic[worst_global_idx]) > 1e-10:
                sign = np.sign(np.real(np.conj(ac_periodic[worst_global_idx]) * d_ac_dc))
                coeffs[k] -= step * sign * np.real(np.conj(ac_periodic[worst_global_idx]) * d_ac_dc)
        
        # Ensure amplitudes stay positive
        amp_check = np.zeros(n_freq)
        for k in range(num_cheb):
            angles = k * np.arccos(t)
            amp_check += coeffs[k] * np.cos(angles)
        
        if np.min(amp_check) < 0.005:
            # Scale back
            coeffs *= 0.9
        
        if max_ac < best_val:
            best_val = max_ac
            best_coeffs = coeffs.copy()
            best_amp = amp.copy()
    
    # Final evaluation
    amp_final = np.zeros(n_freq)
    for k in range(num_cheb):
        angles = k * np.arccos(t)
        amp_final += best_coeffs[k] * np.cos(angles)
    amp_final = np.maximum(0.01, np.abs(amp_final))
    
    return amp_final, best_val

def _phase_retrieval(amp, rng, iters=50):
    """
    Phase retrieval: find a phase that, combined with the given amplitude,
    produces a time-domain signal with good properties.
    """
    n_freq = len(amp)
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    n_freq_fft = L_fft // 2 + 1
    
    # Interpolate amp to FFT size
    freqs_orig = np.linspace(0, 0.5, n_freq)
    freqs_new = np.linspace(0, 0.5, n_freq_fft)
    amp_interp = np.interp(freqs_new, freqs_orig, amp)
    
    # Initialize phase
    phase = rng.uniform(0, 2 * np.pi, size=n_freq_fft)
    phase[0] = 0
    
    best_x = None
    best_val = np.inf
    
    for it in range(iters):
        if _check_deadline():
            break
        
        spectrum = amp_interp * np.exp(1j * phase)
        spectrum[0] = amp_interp[0]
        
        x_full = np.fft.irfft(spectrum, n=L_fft)
        x = x_full[:N]
        
        s = np.sum(x)
        if s < 0:
            x = -x
            phase = -phase
            phase[0] = 0
        
        val = _objective_only(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Update phase: use the phase of the current FFT
        X = np.fft.rfft(x, n=L_fft)
        phase_new = np.angle(X)
        phase_new[0] = 0
        
        # Mix with old phase for stability
        phase = 0.7 * phase_new + 0.3 * phase
        phase[0] = 0
    
    if best_x is None:
        spectrum = amp_interp * np.exp(1j * phase)
        spectrum[0] = amp_interp[0]
        x_full = np.fft.irfft(spectrum, n=L_fft)
        best_x = x_full[:N]
        s = np.sum(best_x)
        if s < 0:
            best_x = -best_x
        best_val = _objective_only(best_x)
    
    return best_x, best_val

def _generate_diverse_envelopes(rng, num_envelopes=20):
    """Generate diverse spectral envelope initializations."""
    n_freq = 256
    freqs = np.linspace(0, 0.5, n_freq)
    
    envelopes = []
    
    # Flat envelope
    envelopes.append(np.ones(n_freq))
    
    # Gaussian envelopes
    for _ in range(5):
        center = rng.uniform(0.05, 0.45)
        width = rng.uniform(0.05, 0.3)
        env = np.exp(-((freqs - center) ** 2) / (2 * width ** 2))
        env = env / np.max(env)
        envelopes.append(env)
    
    # Multi-peak envelopes
    for _ in range(5):
        env = 0.1 * np.ones(n_freq)
        num_peaks = rng.integers(3, 7)
        for _ in range(num_peaks):
            peak_freq = rng.uniform(0.02, 0.48)
            peak_width = rng.uniform(0.01, 0.1)
            peak_height = rng.uniform(0.3, 1.0)
            env += peak_height * np.exp(-((freqs - peak_freq) ** 2) / (2 * peak_width ** 2))
        env = env / np.max(env)
        envelopes.append(env)
    
    # Chebyshev-based envelopes
    for _ in range(5):
        t = 2 * freqs - 1
        t = np.clip(t, -1, 1)
        env = np.ones(n_freq)
        num_terms = rng.integers(5, 15)
        for k in range(num_terms):
            coeff = rng.normal(0, np.exp(-0.2 * k))
            env += coeff * np.cos(k * np.arccos(t))
        env = np.maximum(0.01, np.abs(env))
        env = env / np.max(env)
        envelopes.append(env)
    
    # Tapered envelopes
    for _ in range(3):
        taper_left = rng.uniform(0.01, 0.1)
        taper_right = rng.uniform(0.01, 0.1)
        env = np.minimum(1.0, freqs / taper_left) * np.minimum(1.0, (0.5 - freqs) / taper_right)
        env = env / np.max(env)
        envelopes.append(env)
    
    return envelopes, freqs

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse spectral envelopes and optimize via Remez-like method
    envelopes, freqs = _generate_diverse_envelopes(rng, num_envelopes=20)
    
    for amp_init in envelopes:
        if _check_deadline():
            break
        
        # Optimize envelope via Remez-like exchange
        amp_opt, env_val = _remez_envelope_optimize(freqs, amp_init, num_cheb=15, iters=50)
        
        # Phase retrieval to get time-domain signal
        x_candidate, val_candidate = _phase_retrieval(amp_opt, rng, iters=30)
        
        if val_candidate < best_val:
            best_val = val_candidate
            best_x = x_candidate.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS polish from best solution
    for beta in [20.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Spectral phase optimization
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(best_x, n=L_fft)
    amp = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_phase = phase.copy()
    best_phase_val = best_val
    
    for it in range(30):
        if _check_deadline():
            break
        
        phase_trial = phase + rng.normal(0, 0.1, size=n_freq)
        phase_trial[0] = 0
        
        spectrum = amp * np.exp(1j * phase_trial)
        spectrum[0] = amp[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
        
        val = _objective_only(x_trial)
        if val < best_phase_val:
            best_phase_val = val
            phase = phase_trial.copy()
            best_phase = phase.copy()
    
    spectrum = amp * np.exp(1j * best_phase)
    spectrum[0] = amp[0]
    x_phase = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_phase)
    if s < 0:
        x_phase = -x_phase
    
    if best_phase_val < best_val:
        best_val = best_phase_val
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Final high-beta L-BFGS polish
    for beta in [200.0, 400.0]:
        if _check_deadline():
            break
        
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 150, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Coordinate descent refinement
    step_size = 0.005
    no_improve = 0
    max_no_improve = 3000
    
    while no_improve < max_no_improve and not _check_deadline():
        n_perturb = rng.integers(2, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0005, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 500 == 0:
                step_size = min(0.02, step_size * 1.05)
    
    return best_x
# EVOLVE-BLOCK-END
