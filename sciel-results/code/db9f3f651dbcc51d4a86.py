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
    
    # === Phase 1: Rectangular pulse superposition with differential evolution ===
    # Represent h as sum of disjoint rectangular pulses with variable positions, widths, heights
    def pulses_to_h(params):
        """
        params: [base, (height, center, width), (height, center, width), ...]
        Each pulse is a rectangle of given height centered at center with total width.
        Pulses can overlap; result is clipped to nonnegative.
        """
        h = np.full(N, max(0.0, params[0]))
        for k in range(1, len(params), 3):
            height = params[k]
            center = params[k+1]
            width = params[k+2]
            if width < 1e-8:
                continue
            # Rectangle from center-width/2 to center+width/2
            left = center - width / 2.0
            right = center + width / 2.0
            mask = (X >= left) & (X < right)
            h[mask] += height
        return np.clip(h, 0.0, None)
    
    def pulse_ratio(params):
        h = pulses_to_h(params)
        return ratio_from_h(h)
    
    rng = np.random.default_rng(42)
    
    # Try different numbers of pulses
    n_pulses_options = [2, 3, 4, 5]
    
    for n_p in n_pulses_options:
        if time.time() > deadline - 60.0:
            break
        
        n_starts = 8
        for start in range(n_starts):
            if time.time() > deadline - 55.0:
                break
            
            # Random initial parameters
            base = float(rng.uniform(0.0, 0.2))
            params = np.array([base])
            for _ in range(n_p):
                height = float(rng.uniform(0.5, 5.0))
                center = float(rng.uniform(-0.2, 0.2))
                width = float(rng.uniform(0.01, 0.15))
                params = np.append(params, [height, center, width])
            
            try:
                result = opt.differential_evolution(
                    pulse_ratio,
                    bounds=[(0.0, 1.0)] + [(0.1, 10.0), (-0.25, 0.25), (0.005, 0.2)] * n_p,
                    seed=42 + start,
                    maxiter=30,
                    popsize=15,
                    tol=1e-4,
                    init='sobol'
                )
                if result.fun < best_val:
                    h_opt = pulses_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 2: Fourier-domain optimization ===
    N_HARM = 20
    
    def fourier_to_h(coeffs):
        h = np.full(N, coeffs[0])
        for k in range(1, N_HARM + 1):
            a_k = coeffs[2*k - 1]
            b_k = coeffs[2*k]
            h += a_k * np.cos(2 * np.pi * k * np.arange(N) / N)
            h += b_k * np.sin(2 * np.pi * k * np.arange(N) / N)
        return np.clip(h, 0.0, None)
    
    def fourier_ratio(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)
    
    n_fourier_starts = 15
    
    for start in range(n_fourier_starts):
        if time.time() > deadline - 30.0:
            break
        
        coeffs = np.zeros(2 * N_HARM + 1)
        coeffs[0] = float(rng.uniform(0.5, 2.0))
        for k in range(1, N_HARM + 1):
            coeffs[2*k - 1] = float(rng.uniform(-0.5, 0.5))
            coeffs[2*k] = float(rng.uniform(-0.5, 0.5))
        
        try:
            result = opt.minimize(
                fourier_ratio,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if result.fun < best_val:
                h_opt = fourier_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 3: Coordinate descent refinement ===
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        n_iter = 8000
        
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Pick random coordinates
            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                factor = float(rng2.uniform(0.85, 1.15))
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
