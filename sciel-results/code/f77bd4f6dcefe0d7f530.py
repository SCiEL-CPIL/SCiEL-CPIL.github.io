import numpy as np
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

# Precompute FFT setup
K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def _build_fine_grid(heights):
    """Build fine-sampled function from step heights."""
    return np.repeat(heights, K)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = _build_fine_grid(heights)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    score = max_conv / (integral_f ** 2)
    return score

def _spectral_objective_and_grad(amps, phases, penalty_weight=100.0):
    """
    Objective: minimize max of autoconvolution of |IDFT(amps * exp(i*phases))|^2
    with penalty for non-negativity violations.
    
    amps: array of real amplitudes (length M_FFT//2 + 1)
    phases: array of phases (length M_FFT//2 + 1)
    
    Returns (score, grad_amps, grad_phases)
    """
    # Build complex spectrum
    spec = amps * np.exp(1j * phases)
    
    # Ensure Hermitian symmetry for real output
    # spec[0] and spec[M_FFT//2] should be real
    spec[0] = np.real(spec[0])
    if M_FFT % 2 == 0:
        spec[M_FFT//2] = np.real(spec[M_FFT//2])
    
    # IDFT to get time-domain signal
    f_signal = np.fft.irfft(spec, n=M_FFT)
    
    # Square it to get the step function (approximate)
    h_fine = f_signal ** 2
    
    # Extract step heights by averaging over K samples
    h_steps = np.zeros(N)
    for i in range(N):
        start = i * K
        end = start + K
        if end <= M_STEP:
            h_steps[i] = np.mean(h_fine[start:end])
    
    # Clip to nonnegative
    h_nonneg = np.maximum(h_steps, 0.0)
    
    # Compute autoconvolution
    w = 0.5 / N
    integral_f = np.sum(h_nonneg) * w
    
    if integral_f <= 0:
        return np.inf, np.zeros_like(amps), np.zeros_like(phases)
    
    f_fine_full = np.zeros(M_FFT, dtype=np.float64)
    f_fine_full[:M_STEP] = np.repeat(h_nonneg, K)
    
    F_full = np.fft.rfft(f_fine_full)
    conv = np.fft.irfft(F_full * F_full, n=M_FFT)
    
    max_conv = np.max(conv)
    score = max_conv / (integral_f ** 2)
    
    # Penalty for negative values in h_fine
    neg_mask = h_fine < 0
    penalty = penalty_weight * np.sum(h_fine[neg_mask] ** 2)
    
    total_score = score + penalty
    
    # Gradient computation via automatic differentiation approximation
    # For simplicity, use finite differences for gradient estimation
    eps = 1e-5
    grad_amps = np.zeros_like(amps)
    grad_phases = np.zeros_like(phases)
    
    # Only compute gradients for a subset to save time
    num_grad_points = min(len(amps), 50)
    indices = np.random.choice(len(amps), num_grad_points, replace=False)
    
    for idx in indices:
        # Perturb amplitude
        amps_pert = amps.copy()
        amps_pert[idx] += eps
        _, score_pert, _ = _spectral_eval(amps_pert, phases)
        grad_amps[idx] = (score_pert - total_score) / eps
        
        # Perturb phase
        phases_pert = phases.copy()
        phases_pert[idx] += eps
        _, score_pert, _ = _spectral_eval(amps, phases_pert)
        grad_phases[idx] = (score_pert - total_score) / eps
    
    return total_score, grad_amps, grad_phases

def _spectral_eval(amps, phases):
    """Evaluate spectral objective without gradients."""
    spec = amps * np.exp(1j * phases)
    spec[0] = np.real(spec[0])
    if M_FFT % 2 == 0:
        spec[M_FFT//2] = np.real(spec[M_FFT//2])
    
    f_signal = np.fft.irfft(spec, n=M_FFT)
    h_fine = f_signal ** 2
    
    h_steps = np.zeros(N)
    for i in range(N):
        start = i * K
        end = start + K
        if end <= M_STEP:
            h_steps[i] = np.mean(h_fine[start:end])
    
    h_nonneg = np.maximum(h_steps, 0.0)
    
    w = 0.5 / N
    integral_f = np.sum(h_nonneg) * w
    
    if integral_f <= 0:
        return np.inf, np.inf, np.inf
    
    f_fine_full = np.zeros(M_FFT, dtype=np.float64)
    f_fine_full[:M_STEP] = np.repeat(h_nonneg, K)
    
    F_full = np.fft.rfft(f_fine_full)
    conv = np.fft.irfft(F_full * F_full, n=M_FFT)
    
    max_conv = np.max(conv)
    score = max_conv / (integral_f ** 2)
    
    neg_mask = h_fine < 0
    penalty = 100.0 * np.sum(h_fine[neg_mask] ** 2)
    
    return score + penalty, score, penalty

def _spectral_optimize(initial_amps, initial_phases, max_iter=500):
    """Optimize spectral amplitudes and phases."""
    amps = initial_amps.copy()
    phases = initial_phases.copy()
    
    best_score = np.inf
    best_amps = amps.copy()
    best_phases = phases.copy()
    
    lr = 0.01
    
    for step in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        score, grad_amps, grad_phases = _spectral_objective_and_grad(amps, phases)
        
        if score < best_score:
            best_score = score
            best_amps = amps.copy()
            best_phases = phases.copy()
        
        # Gradient descent
        amps -= lr * grad_amps
        phases -= lr * grad_phases
        
        # Keep amplitudes non-negative
        amps = np.maximum(amps, 0.0)
        
        # Normalize to prevent drift
        if np.max(amps) > 0:
            amps = amps / np.max(amps)
    
    return best_amps, best_phases, best_score

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Strategy 1: Try various smooth shapes as initializations
    x = np.linspace(0, 1, N)
    
    # Raised cosine variations
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
        h_cos = np.maximum(h_cos, 0.0)
        score = compute_score(h_cos)
        if score < best_score:
            best_score = score
            best_h = h_cos.copy()
    
    # Plateau with tapered edges
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        h_plateau = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 0:
            ramp = np.linspace(0, 1, taper_len)
            h_plateau[:taper_len] = ramp
            h_plateau[-taper_len:] = ramp[::-1]
        score = compute_score(h_plateau)
        if score < best_score:
            best_score = score
            best_h = h_plateau.copy()
    
    # Gaussian-like shapes
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        h_gauss = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        h_gauss = h_gauss / np.max(h_gauss)
        score = compute_score(h_gauss)
        if score < best_score:
            best_score = score
            best_h = h_gauss.copy()
    
    # Strategy 2: Random smooth perturbations
    rng = np.random.default_rng(42)
    for trial in range(50):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        g = np.zeros(N)
        num_modes = rng.integers(2, 10)
        for _ in range(num_modes):
            k = rng.integers(1, N // 4)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.05, 0.3)
            g += amp * np.cos(2 * np.pi * k * x + phase)
        
        g -= np.mean(g)
        
        for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            h_trial = 1.0 + eps * g
            h_trial = np.maximum(h_trial, 0.0)
            if np.sum(h_trial) <= 0:
                continue
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # Strategy 3: Spectral amplitude optimization
    if time.time() - START_TIME < TIME_LIMIT - 25:
        # Initialize from best_h
        f_fine_init = np.zeros(M_FFT, dtype=np.float64)
        f_fine_init[:M_STEP] = _build_fine_grid(best_h)
        F_init = np.fft.rfft(f_fine_init)
        
        amps_init = np.abs(F_init)
        phases_init = np.angle(F_init)
        
        # Optimize
        best_amps, best_phases, spec_score = _spectral_optimize(amps_init, phases_init, max_iter=200)
        
        # Convert back to step function
        spec = best_amps * np.exp(1j * best_phases)
        spec[0] = np.real(spec[0])
        if M_FFT % 2 == 0:
            spec[M_FFT//2] = np.real(spec[M_FFT//2])
        
        f_signal = np.fft.irfft(spec, n=M_FFT)
        h_fine = np.maximum(f_signal ** 2, 0.0)
        
        h_steps = np.zeros(N)
        for i in range(N):
            start = i * K
            end = start + K
            if end <= M_STEP:
                h_steps[i] = np.mean(h_fine[start:end])
        
        h_steps = np.maximum(h_steps, 0.0)
        
        score = compute_score(h_steps)
        if score < best_score:
            best_score = score
            best_h = h_steps.copy()
    
    # Strategy 4: Final local search
    if time.time() - START_TIME < TIME_LIMIT - 5:
        for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            h_scaled = best_h * scale
            score = compute_score(h_scaled)
            if score < best_score:
                best_score = score
                best_h = h_scaled.copy()
        
        rng2 = np.random.default_rng(123)
        for _ in range(100):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng2.normal(0, 0.005, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
