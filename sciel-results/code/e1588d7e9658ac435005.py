"""Model-and-optimize: epigraph formulation + multi-restart L-BFGS + spectral polish + coordinate descent."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 2)

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
    Uses numerically stable log-sum-exp.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Numerically stable log-sum-exp
    shifted = abs_ac - m
    exps = np.exp(beta * shifted)
    sum_exps = np.sum(exps)
    
    if sum_exps < 1e-300:
        soft_val = m
        weights = np.ones(len(abs_ac)) / len(abs_ac)
    else:
        log_sum = np.log(sum_exps)
        soft_val = m + log_sum / beta
        weights = exps / sum_exps
    
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient with numerical safeguards."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    s2 = s * s
    s3 = s2 * s
    val = soft_val / s2
    grad = grad_soft / s2 - 2.0 * soft_val * np.ones(N) / s3
    # Clip gradient to prevent numerical explosion
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e6:
        grad = grad * (1e6 / grad_norm)
    return val, grad

def _epigraph_objective_and_grad(z, t, beta):
    """
    Epigraph formulation: minimize t subject to |ac[k]| <= t * (sum x)^2 for all k.
    Penalty approach: minimize t + lambda * soft_max(|ac|/s^2 - t, beta)
    
    Variables: x (N), t (scalar)
    """
    x = z[:N]
    t_val = z[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N + 1)
    
    s2 = s * s
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac) / s2
    
    # soft_max of (abs_ac - t)
    shifted = abs_ac - t_val
    m = np.max(shifted)
    
    if m < -1e-8:
        soft_pen = 0.0
        grad_x = np.zeros(N)
        grad_t = 1.0
    else:
        exps = np.exp(beta * (shifted - m))
        log_sum = np.log(np.sum(exps))
        soft_pen = m + log_sum / beta
        weights = exps / np.sum(exps)
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        
        # d(soft_pen)/dx[n] via FFT cross-correlation
        L_corr = 1
        while L_corr < 4 * N:
            L_corr *= 2
        g_padded = np.zeros(L_corr)
        g_padded[:2*N-1] = weights * signs
        X_f = np.fft.rfft(x, n=L_corr)
        G_f = np.fft.rfft(g_padded, n=L_corr)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
        
        # Gradient of soft_pen w.r.t. x:
        # d/dx[n] soft_max_k(|ac[k]|/s^2 - t)
        # = sum_k w[k] * sign(ac[k]) * d(ac[k])/dx[n] / s^2
        #   - sum_k w[k] * sign(ac[k]) * ac[k] * 2 / s^3
        # First term: 2 * corr[n] / s^2
        # Second term: -2 * (sum_k w[k]*sign(ac[k])*ac[k]) / s^3
        weighted_ac = np.sum(weights * signs * ac)
        grad_x = 2.0 * corr[:N] / s2 - 2.0 * weighted_ac * np.ones(N) / (s2 * s)
        
        # d(soft_pen)/dt = -sum_k w[k] = -1
        grad_t = 1.0 - 1.0  # t + penalty*(-1), so grad_t = 1 - lambda
    
    lam = 50.0  # penalty weight
    obj_val = t_val + lam * soft_pen
    grad = np.concatenate([lam * grad_x, [1.0 - lam]])
    
    return obj_val, grad

def _spectral_init(rng, mode=0):
    """Generate initial x via spectral phase retrieval."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_fft // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=150, lr=0.01):
    """Optimize the phase of the spectrum while keeping magnitude fixed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 50.0)
        G = np.fft.rfft(grad, n=L_fft)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _amplitude_optimize(x0, rng, iters=30):
    """Optimize the amplitude spectrum while keeping phase fixed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    phase = np.angle(X)
    phase[0] = 0
    amp = np.abs(X).copy()
    
    best_amp = amp.copy()
    best_val = _objective_only(x0)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        amp_trial = amp * (1 + 0.05 * rng.normal(0, 1, size=n_freq))
        amp_trial = np.maximum(0.01, amp_trial)
        
        spectrum = amp_trial * np.exp(1j * phase)
        spectrum[0] = amp_trial[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            amp = amp_trial.copy()
            best_amp = amp.copy()
    
    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _spectral_coord_descent(x0, rng, max_iters=8000, initial_step=0.05):
    """
    Spectral-domain coordinate descent: optimize individual frequency bins
    of the power spectrum (PSD) to minimize peak autoconvolution.
    Uses batch updates for efficiency.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    phase = np.angle(X)
    amp = np.abs(X).copy()
    
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step_size = initial_step
    no_improve = 0
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Pick a random frequency bin to perturb
        k = rng.integers(1, n_freq - 1)  # skip DC and Nyquist
        
        # Try small perturbations to this bin's amplitude
        best_local_amp = amp[k]
        best_local_val = best_val
        
        for scale in [0.85, 0.95, 1.05, 1.15]:
            amp_trial = amp.copy()
            amp_trial[k] = amp[k] * scale
            
            spectrum = amp_trial * np.exp(1j * phase)
            spectrum[0] = amp_trial[0]
            x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
            
            s = np.sum(x_trial)
            if s < 0:
                x_trial = -x_trial
            
            val = _objective_only(x_trial)
            if val < best_local_val:
                best_local_val = val
                best_local_amp = amp_trial[k]
        
        if best_local_val < best_val:
            best_val = best_local_val
            amp[k] = best_local_amp
            # Reconstruct x
            spectrum = amp * np.exp(1j * phase)
            spectrum[0] = amp[0]
            best_x = np.fft.irfft(spectrum, n=L_fft)[:N]
            s = np.sum(best_x)
            if s < 0:
                best_x = -best_x
            step_size = max(0.01, step_size * 0.98)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.1, step_size * 1.01)
        
        # Occasionally perturb phase instead
        if rng.random() < 0.2:
            k = rng.integers(1, n_freq - 1)
            dphase = step_size * rng.normal(0, 1)
            phase_trial = phase.copy()
            phase_trial[k] += dphase
            
            spectrum = amp * np.exp(1j * phase_trial)
            spectrum[0] = amp[0]
            x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
            
            s = np.sum(x_trial)
            if s < 0:
                x_trial = -x_trial
                phase_trial = -phase_trial
                phase_trial[0] = 0
            
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                phase[k] = phase_trial[k]
                spectrum = amp * np.exp(1j * phase)
                spectrum[0] = amp[0]
                best_x = np.fft.irfft(spectrum, n=L_fft)[:N]
                s = np.sum(best_x)
                if s < 0:
                    best_x = -best_x
                no_improve = 0
            else:
                no_improve += 1
    
    return best_x, best_val

def _peak_specific_softmax_grad(x, peak_lags, beta):
    """
    Compute gradient of soft-max over only the peak lags.
    This focuses optimization effort on the actual bottleneck lags.
    """
    ac = _autoconv_fft(x)
    s = np.sum(x)
    s2 = s * s
    
    abs_ac_peak = np.abs(ac[peak_lags])
    m = np.max(abs_ac_peak)
    shifted = abs_ac_peak - m
    exps = np.exp(beta * shifted)
    sum_exps = np.sum(exps)
    
    if sum_exps < 1e-300:
        weights = np.ones(len(peak_lags)) / len(peak_lags)
    else:
        weights = exps / sum_exps
    
    signs = np.sign(ac[peak_lags])
    signs[ac[peak_lags] == 0] = 1.0
    
    # Build gradient vector for these lags
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    for i, lag in enumerate(peak_lags):
        g_padded[int(lag)] = weights[i] * signs[i]
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    
    # Gradient of soft_max(|ac[peak_lags]|/s^2) w.r.t. x
    # = sum_i w_i * sign(ac[i]) * d(ac[i])/dx[n] / s^2
    #   - sum_i w_i * sign(ac[i]) * ac[i] * 2 / s^3
    weighted_ac = np.sum(weights * signs * ac[peak_lags])
    grad_x = 2.0 * corr[:N] / s2 - 2.0 * weighted_ac * np.ones(N) / (s2 * s)
    
    return grad_x

def _peak_targeted_polish(x0, rng, max_iters=5000):
    """
    Targeted polish: identify the lag(s) contributing to the peak autoconvolution
    and apply targeted adjustments to reduce them using peak-specific gradient.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        s2 = s * s
        abs_ac = np.abs(ac) / s2
        
        # Find the top few lags contributing to the peak
        top_k = 5
        peak_lags = np.argsort(abs_ac)[-top_k:]
        
        # Use peak-specific gradient for descent
        beta = 50.0
        grad_peak = _peak_specific_softmax_grad(best_x, peak_lags, beta)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_peak)
        if grad_norm > 1e-10:
            grad_peak = grad_peak / grad_norm
        
        # Try a few step sizes
        improved = False
        for step in [0.02, 0.01, 0.005, 0.002]:
            x_trial = best_x - step * grad_peak
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                improved = True
                break
        
        if not improved:
            # Try sign-flip on elements near peak lags
            n_flip = rng.integers(2, 6)
            indices = rng.choice(N, size=n_flip, replace=False)
            x_trial = best_x.copy()
            x_trial[indices] = -x_trial[indices] * 0.8
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
            else:
                # Random perturbation
                if rng.random() < 0.1:
                    x_trial = best_x + rng.normal(0, 0.02, size=N)
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial.copy()
    
    return best_x, best_val

def _coordinate_descent(x0, rng, max_iters=12000, initial_step=0.02):
    """Coordinate descent with structured perturbations and peak-aware moves."""
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step_size = initial_step
    no_improve = 0
    
    # Track peak lags for peak-aware perturbations
    peak_lags_cache = None
    last_peak_update = 0
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Update peak lags every 50 iterations
        if it - last_peak_update >= 50 or peak_lags_cache is None:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
            peak_lags_cache = np.argsort(abs_ac)[-5:]
            last_peak_update = it
        
        r = rng.random()
        if r < 0.20:
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.40:
            n_perturb = rng.integers(2, 8)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.55:
            freq = rng.integers(1, 8)
            phase = rng.uniform(0, 2 * np.pi)
            t_arr = np.arange(N)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        elif r < 0.70:
            n_perturb = rng.integers(5, 20)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size * 2, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.80:
            # Peak-aware: perturb elements near peak lags
            lag = int(peak_lags_cache[rng.integers(0, len(peak_lags_cache))])
            if lag < N:
                idx = int(rng.integers(max(0, lag-3), min(N, lag+4)))
                delta = step_size * 1.5 * rng.choice([-1, 1])
                x_trial = best_x.copy()
                x_trial[idx] += delta
            else:
                idx = rng.integers(0, N)
                x_trial = best_x.copy()
                x_trial[idx] += step_size
        elif r < 0.90:
            # Sign-flip on a random subset
            n_flip = rng.integers(3, 15)
            indices = rng.choice(N, size=n_flip, replace=False)
            x_trial = best_x.copy()
            x_trial[indices] = -x_trial[indices]
            x_trial[indices] *= 0.7
        else:
            # Pairwise swap: flip signs of two nearby elements
            idx1 = rng.integers(0, N)
            idx2 = int(rng.integers(max(0, idx1-5), min(N, idx1+6)))
            x_trial = best_x.copy()
            x_trial[idx1] = -x_trial[idx1]
            x_trial[idx2] = -x_trial[idx2]
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.05, step_size * 1.015)
        
        if no_improve > 2000 and rng.random() < 0.02:
            x_restart = best_x + rng.normal(0, 0.05, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart.copy()
                no_improve = 0
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
        # Additional diverse initializations
        1.0 + 0.7 * np.sin(2 * np.pi * 5 * t),
        1.0 + 0.6 * np.cos(2 * np.pi * 7 * t),
        np.maximum(0.2, 1.0 + 0.8 * np.sin(2 * np.pi * 2 * t) * np.cos(2 * np.pi * 3 * t)),
        1.0 - 0.5 * np.cos(4 * np.pi * t),
    ]
    
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(8):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Add sign-symmetric patterns (exploiting sign-flip invariance)
    for _ in range(5):
        base = rng.normal(1.0, 0.3, size=N)
        # Make it sign-symmetric: x[i] = -x[N-1-i]
        sym = np.zeros(N)
        for i in range(N // 2):
            sym[i] = base[i]
            sym[N - 1 - i] = -base[i]
        if N % 2 == 1:
            sym[N // 2] = 0
        initial_points.append(sym)
    
    # Phase 1: Multi-restart L-BFGS with soft-max
    betas = [40.0, 80.0, 150.0]
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 200, 'gtol': 1e-9, 'ftol': 1e-12, 'maxls': 40})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Epigraph formulation optimization with adaptive beta
    for beta_epi in [50.0, 100.0, 200.0, 400.0]:
        if _check_deadline():
            break
        
        x_start = best_x + rng.normal(0, 0.02, size=N)
        t_init = _objective_only(x_start) * 1.01
        
        z0 = np.concatenate([x_start, [t_init]])
        
        try:
            def epi_func(z, _beta=beta_epi):
                return _epigraph_objective_and_grad(z, 0, _beta)
            
            result = minimize(epi_func, z0, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 40})
            
            x_epigraph = result.x[:N]
            val_epigraph = _objective_only(x_epigraph)
            
            if val_epigraph < best_val:
                best_val = val_epigraph
                best_x = x_epigraph.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=150, lr=0.015)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Amplitude optimization
    for _ in range(4):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=25)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: High-beta L-BFGS polish
    for beta in [250.0, 500.0, 1000.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 130, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 25})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Spectral coordinate descent (PSD-domain optimization)
    if not _check_deadline():
        best_x, best_val = _spectral_coord_descent(best_x, rng, max_iters=6000, initial_step=0.05)
    
    # Phase 7: Time-domain coordinate descent refinement
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, rng, max_iters=10000, initial_step=0.02)
    
    # Phase 8: Final high-beta polish after coordinate descent
    if not _check_deadline():
        for beta in [400.0, 800.0, 1500.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 120, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 25})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    # Phase 9: Peak-targeted polish
    if not _check_deadline():
        best_x, best_val = _peak_targeted_polish(best_x, rng, max_iters=3000)
    
    # Phase 10: One more spectral polish if time permits
    if not _check_deadline():
        best_x, best_val = _spectral_coord_descent(best_x, rng, max_iters=2000, initial_step=0.03)
    
    # Phase 11: Final high-precision polish with very high beta
    if not _check_deadline():
        for beta_final in [2000.0, 5000.0, 10000.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_final(x, _beta=beta_final):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_final, best_x, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 20})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    # Phase 12: Final coordinate descent with tiny steps
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, rng, max_iters=5000, initial_step=0.005)
    
    # Phase 13: Final peak-specific gradient descent
    if not _check_deadline():
        best_x_final = best_x.copy()
        best_val_final = best_val
        for _ in range(2000):
            if _check_deadline():
                break
            ac = _autoconv_fft(best_x_final)
            abs_ac = np.abs(ac)
            peak_lags = np.argsort(abs_ac)[-5:]
            grad_peak = _peak_specific_softmax_grad(best_x_final, peak_lags, 80.0)
            grad_norm = np.linalg.norm(grad_peak)
            if grad_norm < 1e-12:
                break
            grad_peak = grad_peak / grad_norm
            improved = False
            for step in [0.01, 0.005, 0.002, 0.001]:
                x_trial = best_x_final - step * grad_peak
                val = _objective_only(x_trial)
                if val < best_val_final:
                    best_val_final = val
                    best_x_final = x_trial.copy()
                    improved = True
                    break
            if not improved and rng.random() < 0.05:
                x_trial = best_x_final + rng.normal(0, 0.005, size=N)
                val = _objective_only(x_trial)
                if val < best_val_final:
                    best_val_final = val
                    best_x_final = x_trial.copy()
        if best_val_final < best_val:
            best_val = best_val_final
            best_x = best_x_final.copy()
    
    return best_x
# EVOLVE-BLOCK-END
