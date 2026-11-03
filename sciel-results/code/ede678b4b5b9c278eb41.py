import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def quick_ratio(h):
    """Fast ratio computation for coordinate descent."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Complex-signal magnitude representation ===
    # h(x) = |sum_k a_k * exp(i*(k*omega*x + phi_k))|^2
    # This is always non-negative and has tunable oscillatory structure
    # Parameters: for each mode k: amplitude a_k, phase phi_k, and a global scale
    
    def complex_signal_to_h(params, n_modes):
        """Convert complex signal parameters to non-negative step function."""
        # params: [scale, a1, phi1, a2, phi2, ..., an, phin]
        scale = params[0]
        h = np.zeros(N)
        for k in range(n_modes):
            a_k = params[1 + 2*k]
            phi_k = params[2 + 2*k]
            # Use different frequencies for each mode
            freq = (k + 1) * 2.0 * np.pi / 0.5  # frequency in rad per unit x
            # Actually, let's use harmonics of the fundamental
            freq = (k + 1) * 2.0 * np.pi * 2.0  # 2 cycles per unit interval
            h += a_k * np.cos(freq * X + phi_k)
        # h is the real part; we want |complex|^2
        # Better: build complex signal then take magnitude squared
        S = np.zeros(N, dtype=complex)
        for k in range(n_modes):
            a_k = params[1 + 2*k]
            phi_k = params[2 + 2*k]
            freq = (k + 1) * 2.0 * np.pi * 2.0
            S += a_k * np.exp(1j * (freq * X + phi_k))
        h = np.abs(S) ** 2
        h *= scale
        return h
    
    def complex_ratio(params, n_modes):
        h = complex_signal_to_h(params, n_modes)
        return ratio_from_h(h)
    
    # Try different numbers of modes
    for n_modes in [2, 3, 4]:
        n_starts = 12
        for start in range(n_starts):
            if time.time() > deadline - 35.0:
                break
            
            # Initialize parameters
            scale = float(rng.uniform(0.1, 1.0))
            params = np.array([scale])
            for k in range(n_modes):
                a_k = float(rng.uniform(0.1, 2.0))
                phi_k = float(rng.uniform(-np.pi, np.pi))
                params = np.append(params, [a_k, phi_k])
            
            try:
                result = opt.minimize(
                    lambda p: complex_ratio(p, n_modes),
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if result.fun < best_val:
                    h_opt = complex_signal_to_h(result.x, n_modes)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 2: Phase-only optimization with fixed amplitudes ===
    # Fix amplitudes, optimize only phases and scale
    def phase_only_to_h(params, n_modes, amps):
        scale = params[0]
        S = np.zeros(N, dtype=complex)
        for k in range(n_modes):
            phi_k = params[1 + k]
            freq = (k + 1) * 2.0 * np.pi * 2.0
            S += amps[k] * np.exp(1j * (freq * X + phi_k))
        h = np.abs(S) ** 2
        h *= scale
        return h
    
    def phase_only_ratio(params, n_modes, amps):
        h = phase_only_to_h(params, n_modes, amps)
        return ratio_from_h(h)
    
    # Try a few fixed amplitude patterns
    amp_patterns = [
        [1.0, 0.5, 0.25],
        [1.0, 0.7, 0.4],
        [0.8, 0.8, 0.8],
        [1.0, 0.3, 0.1],
    ]
    
    for amps in amp_patterns:
        n_modes = len(amps)
        n_starts = 8
        for start in range(n_starts):
            if time.time() > deadline - 20.0:
                break
            
            scale = float(rng.uniform(0.1, 1.0))
            params = np.array([scale])
            for k in range(n_modes):
                phi_k = float(rng.uniform(-np.pi, np.pi))
                params = np.append(params, [phi_k])
            
            try:
                result = opt.minimize(
                    lambda p: phase_only_ratio(p, n_modes, amps),
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if result.fun < best_val:
                    h_opt = phase_only_to_h(result.x, n_modes, amps)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 3: Coordinate descent on the best result ===
    if time.time() < deadline - 8.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        step_scales = [1.0, 0.5, 0.25, 0.1]
        n_iter = 15000
        
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            
            scale_idx = min(it // 4000, len(step_scales) - 1)
            scale = step_scales[scale_idx]
            
            n_coords = int(rng2.integers(1, 12))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                factor_range = (1.0 - 0.2 * scale, 1.0 + 0.2 * scale)
                factor = float(rng2.uniform(*factor_range))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 4: Hand-crafted shapes ===
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-2.0 * x_norm**2),
            np.exp(-0.5 * x_norm**2),
            0.5 + 0.5 * np.exp(-8.0 * x_norm**2),
            np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2),
            0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2),
            1.0 - 0.3 * np.exp(-5.0 * x_norm**2),
            0.5 + 0.5 * x_norm**2,
            0.5 + 0.5 * np.cos(np.pi * x_norm),
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm - 0.5)**2),
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm + 0.5)**2),
        ]
        
        for h in crafted:
            if time.time() > deadline - 2.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
