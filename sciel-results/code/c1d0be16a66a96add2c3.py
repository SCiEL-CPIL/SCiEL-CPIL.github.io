"""Combined strategy: random-walk + structured patterns + soft-max + FFT autoconv + L-BFGS-B refinement."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = 2 * N - 1
    H = rfft(h, n=n)
    c = irfft(H * H, n=n)
    return c

def _soft_max(x, beta):
    """Differentiable approximation to max using log-sum-exp."""
    m = np.max(x)
    x_shifted = x - m
    sum_exp = np.sum(np.exp(beta * x_shifted))
    if sum_exp == 0:
        return m
    return m + (1.0 / beta) * np.log(sum_exp)

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _random_walk_to_h(drift, noise_scale, seed=42):
    """
    Generate h from a random walk with drift.
    h[i] = cumsum(noise[j] - drift), then centered + offset.
    """
    rng_local = np.random.default_rng(seed)
    steps = rng_local.standard_normal(N) * noise_scale - drift
    h = np.cumsum(steps)
    h = h - np.mean(h)
    h = h + 0.01
    return h

def _structured_pattern_to_h(freq, phase, amp_ratio, seed=0):
    """
    Generate h from a structured pattern: base + freq*cos(2*pi*freq*i/N + phase) with sign flips.
    amp_ratio controls the amplitude of the oscillation relative to base.
    """
    t = np.arange(N)
    base = 1.0
    osc = amp_ratio * np.cos(2 * np.pi * freq * t / N + phase)
    h = base + osc
    # Add random sign flips to create more complex patterns
    if seed != 0:
        rng_local = np.random.default_rng(seed)
        signs = rng_local.choice([-1, 1], size=N)
        h = h * signs
    return h

def _objective_full(params, beta):
    """Soft-max objective for full h vector."""
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def _objective_walk(params, beta):
    """Objective for random walk parameterization: params = [drift, noise_scale]."""
    drift = params[0]
    noise_scale = params[1]
    h = _random_walk_to_h(drift, noise_scale)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def _objective_structured(params, beta):
    """Objective for structured pattern: params = [freq, phase, amp_ratio, seed_choice]."""
    freq = params[0]
    phase = params[1]
    amp_ratio = params[2]
    seed_choice = int(params[3]) if abs(params[3]) > 0.5 else 0
    h = _structured_pattern_to_h(freq, phase, amp_ratio, seed=seed_choice)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    sm = _soft_max(c, beta)
    val = 2.0 * N * sm / (s * s)
    return val

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # should be 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val
    
    # === Phase 1: Structured pattern exploration ===
    # Try various frequencies, phases, and amplitude ratios
    freqs = [1, 2, 3, 4, 5, 7, 10, 13, 20, 30]
    phases = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    amp_ratios = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    seeds = [0, 1, 2, 3]
    
    for freq in freqs:
        for phase in phases:
            for amp in amp_ratios:
                if time.time() > deadline - 50.0:
                    break
                for seed in seeds:
                    h = _structured_pattern_to_h(freq, phase, amp, seed=seed)
                    try_h(h)
    
    # === Phase 2: Random walk exploration ===
    drifts = [-0.1, -0.05, 0.0, 0.05, 0.1]
    noise_scales = [0.5, 1.0, 1.5, 2.0, 3.0]
    walk_seeds = list(range(20))
    
    for drift in drifts:
        for ns in noise_scales:
            if time.time() > deadline - 50.0:
                break
            for ws in walk_seeds:
                h = _random_walk_to_h(drift, ns, seed=ws)
                try_h(h)
    
    # === Phase 3: Optimize random walk parameters ===
    for beta in [50.0, 200.0, 500.0]:
        if time.time() > deadline - 40.0:
            break
        # Try several initial conditions
        for d0 in [-0.05, 0.0, 0.05]:
            for n0 in [0.5, 1.0, 1.5]:
                if time.time() > deadline - 40.0:
                    break
                try:
                    res = minimize(
                        lambda p: _objective_walk(p, beta),
                        np.array([d0, n0]),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12},
                        bounds=[(-1.0, 1.0), (0.1, 5.0)]
                    )
                    h_opt = _random_walk_to_h(res.x[0], res.x[1])
                    val_opt = _evaluate_exact(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    # === Phase 4: Full-vector L-BFGS-B refinement with soft-max ===
    for beta in [100.0, 500.0, 1000.0, 2000.0]:
        if time.time() > deadline - 25.0:
            break
        
        for restart in range(8):
            if time.time() > deadline - 25.0:
                break
            
            if restart == 0:
                x0 = best_h.copy()
            elif restart == 1:
                noise_scale = 0.02 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 2:
                noise_scale = 0.05 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 3:
                noise_scale = 0.10 * np.std(best_h)
                x0 = best_h + rng.standard_normal(N) * noise_scale
            elif restart == 4:
                mask = rng.random(N) < 0.3
                x0 = best_h.copy()
                x0[mask] += rng.standard_normal(np.sum(mask)) * 0.05 * np.std(best_h)
            elif restart == 5:
                # Structured restart with sign changes
                x0 = _structured_pattern_to_h(
                    freq=rng.integers(1, 20),
                    phase=rng.random() * 2 * np.pi,
                    amp_ratio=rng.uniform(0.5, 1.5),
                    seed=rng.integers(0, 100)
                )
            elif restart == 6:
                # Random walk restart
                x0 = _random_walk_to_h(
                    drift=rng.uniform(-0.1, 0.1),
                    noise_scale=rng.uniform(0.5, 2.0),
                    seed=rng.integers(0, 1000)
                )
            else:
                # Sparse spike restart
                x0 = np.ones(N)
                n_spikes = rng.integers(5, 20)
                spike_idx = rng.integers(0, N, size=n_spikes)
                spike_amps = rng.standard_normal(n_spikes) * 2.0
                np.add.at(x0, spike_idx, spike_amps)
            
            try:
                res = minimize(
                    lambda x: _objective_full(x, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = res.x
                val_opt = _evaluate_exact(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # === Phase 5: Adaptive coordinate descent polish on exact objective ===
    if time.time() < deadline - 8.0:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        improved = True
        while improved and (time.time() < deadline - 8.0):
            improved = False
            for _ in range(300):
                if time.time() > deadline - 8.0:
                    break
                idx = rng.integers(0, N)
                for delta in step_sizes:
                    for sign in [1, -1]:
                        h_new = best_h.copy()
                        h_new[idx] += sign * delta
                        val_new = _evaluate_exact(h_new)
                        if val_new < best_val - 1e-14:
                            best_val = val_new
                            best_h = h_new
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                step_sizes = [s * 0.5 for s in step_sizes]
                if step_sizes[0] < 1e-6:
                    break
    
    # === Phase 6: Final subgradient polish ===
    if time.time() < deadline - 3.0:
        for _ in range(50):
            if time.time() > deadline - 3.0:
                break
            h = best_h.copy()
            s = np.sum(h)
            if abs(s) < 1e-10:
                break
            c = _autoconv_fft(h)
            max_idx = np.argmax(c)
            h = h - np.mean(h) + 1.0
            
            for scale in [0.01, 0.005, 0.001]:
                peak_lag = max_idx
                if peak_lag < N:
                    direction = np.zeros(N)
                    for i in range(N - peak_lag):
                        direction[i] -= h[i + peak_lag]
                        direction[i + peak_lag] -= h[i]
                else:
                    lag = peak_lag - (2*N - 2)
                    direction = np.zeros(N)
                    for i in range(N + lag):
                        direction[i] -= h[i - lag]
                        direction[i - lag] -= h[i]
                
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    h_new = h + scale * direction
                    if abs(np.sum(h_new)) < 1e-10:
                        continue
                    val_new = _evaluate_exact(h_new)
                    if val_new < best_val - 1e-14:
                        best_val = val_new
                        best_h = h_new.copy()
                        break
    
    # Safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
