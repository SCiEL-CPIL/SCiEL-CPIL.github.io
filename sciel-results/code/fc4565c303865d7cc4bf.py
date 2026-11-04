import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 115.0  # safety margin before 150s kill
_start = time.time()

def _elapsed():
    return time.time() - _start

def _time_left():
    return _DEADLINE - _elapsed()

def _autoconv_peak(h):
    """Compute supremum norm of autoconvolution using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return np.max(conv)

def _ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15) or np.sum(h) < 1e-15:
        return 0.0
    h = np.maximum(h, 0)
    l2sq = np.sum(h * h)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if l1 < 1e-15 or peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)

def _shape_score(h):
    """Scale-invariant shape quality: l2_sq / (l1 * l_inf) for the shape itself."""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _autoconvolution(h):
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-30:
        return 1e-30
    x_shifted = x - x_max
    return x_max + sigma * np.log(np.sum(np.exp(x_shifted / sigma))) / len(x)

def _objective_neg_shape_score(x):
    """Minimize negative shape score. h = max(x, 0)."""
    h = np.maximum(x, 0)
    if np.sum(h) < 1e-15:
        return 1e10
    return -_shape_score(h)

def _objective_g(g_flat):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    h_norm = h / l1
    
    l2sq = np.sum(h_norm * h_norm)
    
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h_norm, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    peak = _smooth_max(conv, sigma=0.05)
    
    if peak < 1e-30:
        return 1e10
    
    return -l2sq / peak

def _optimize_from_g(g0, max_iters=300, maxfun=1500):
    """Run L-BFGS-B optimization on complex parameters."""
    result = minimize(
        _objective_g,
        g0,
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxfun}
    )
    g_opt = result.x.reshape(N, 2)
    h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
    h_opt = np.maximum(h_opt, 0)
    return h_opt

def _h_to_g(h):
    """Convert h to g (real part = sqrt(h), imag = 0)."""
    h = np.maximum(h, 0)
    g_real = np.sqrt(h)
    g_imag = np.zeros(N)
    return np.concatenate([g_real, g_imag])

def _sidon_set(n, max_size):
    """Generate a Sidon set (B2 set) in {0,...,n-1} of given size."""
    S = []
    for i in range(n):
        candidate = i
        valid = True
        for j in range(len(S)):
            for k in range(j+1, len(S)):
                s1 = S[j] + S[k]
                s2 = S[j] + candidate
                s3 = S[k] + candidate
                if s1 == s2 or s1 == s3 or s2 == s3:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(candidate)
            if len(S) >= max_size:
                break
    return S

def _optimize_direct(h0, max_iters=500, maxfun=2000):
    """Run L-BFGS-B optimization directly on h space with h = max(x, 0)."""
    result = minimize(
        _objective_neg_shape_score,
        h0,
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxfun}
    )
    h_opt = np.maximum(result.x, 0)
    return h_opt

def _dft_spectral_optimize(K=3, max_iters=200, maxfun=1000):
    """
    Optimize over sparse DFT representation.
    h = |IDFT(z)|^2 where z has only K active bins.
    Parameters: magnitudes and phases of K bins.
    """
    def _objective_spectral(params):
        # params: [mag_1, phase_1, mag_2, phase_2, ..., mag_K, phase_K]
        z = np.zeros(N, dtype=complex)
        for i in range(K):
            mag = params[2*i]
            phase = params[2*i+1]
            z[i] = mag * np.exp(1j * phase)
        
        # IDFT to get h
        h = np.abs(np.fft.ifft(z)) ** 2
        h = np.maximum(h, 0)
        
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 1e10
        h_norm = h / l1
        
        l2sq = np.sum(h_norm * h_norm)
        
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h_norm, n_fft)
        conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
        peak = _smooth_max(conv, sigma=0.05)
        
        if peak < 1e-30:
            return 1e10
        
        return -l2sq / peak
    
    # Initialize with random magnitudes and phases
    rng = np.random.RandomState(42)
    params0 = np.zeros(2 * K)
    for i in range(K):
        params0[2*i] = 1.0  # magnitude
        params0[2*i+1] = rng.uniform(0, 2 * np.pi)  # phase
    
    result = minimize(
        _objective_spectral,
        params0,
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxfun}
    )
    
    # Extract h from optimal parameters
    z = np.zeros(N, dtype=complex)
    for i in range(K):
        mag = result.x[2*i]
        phase = result.x[2*i+1]
        z[i] = mag * np.exp(1j * phase)
    
    h = np.abs(np.fft.ifft(z)) ** 2
    h = np.maximum(h, 0)
    return h

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Generate diverse initializations
    initializations_h = []
    
    # 1. Uniform
    initializations_h.append(np.ones(N))
    
    # 2. Gaussian-like profiles
    x = np.linspace(-1, 1, N)
    for var in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        initializations_h.append(np.exp(-x**2 / var))
    
    # 3. Bimodal
    for sep in [0.3, 0.5, 0.7]:
        initializations_h.append(np.exp(-((x - sep)**2) / 0.1) + np.exp(-((x + sep)**2) / 0.1))
    
    # 4. Triangle / tent
    initializations_h.append(1 - np.abs(x))
    
    # 5. Cosine window
    initializations_h.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    # 6. Sidon-set sparse spikes (key structural insight)
    for sidon_size in [3, 4, 5, 6, 7, 8]:
        S = _sidon_set(N, sidon_size)
        if len(S) >= 2:
            h_sidon = np.zeros(N)
            h_sidon[S] = 1.0
            initializations_h.append(h_sidon)
    
    # 7. Random smooth
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations_h.append(h_smooth)
    
    # 8. Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    initializations_h.append(h_center)
    
    # 9. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    initializations_h.append(h_edge)
    
    # 10. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    initializations_h.append(h_alt)
    
    # 11. Linear ramps
    initializations_h.append(np.linspace(0.1, 1.0, N))
    initializations_h.append(np.linspace(1.0, 0.1, N))
    
    # 12. Super-Gaussian
    for p in [2.0, 3.0, 4.0, 6.0]:
        initializations_h.append(np.maximum(0, 1 - np.abs(x) ** p))
    
    # 13. Trapezoidal
    for box_frac in [0.3, 0.5, 0.7]:
        h_trap = np.zeros(N)
        n_box = int(N * box_frac)
        n_taper = (N - n_box) // 2
        idx = n_taper
        h_trap[idx:idx + n_box] = 1.0
        if n_taper > 0:
            h_trap[:n_taper] = np.linspace(0, 1, n_taper)
            h_trap[-n_taper:] = np.linspace(1, 0, n_taper)
        initializations_h.append(h_trap)
    
    # 14. Two-bump
    for sigma in [0.1, 0.2, 0.3]:
        for sep in [0.3, 0.5, 0.7]:
            h_two = (np.exp(-0.5 * ((x - sep/2) / sigma) ** 2) +
                     np.exp(-0.5 * ((x + sep/2) / sigma) ** 2))
            initializations_h.append(h_two)
    
    # Try each initialization with both direct and g-space optimization
    for h_init in initializations_h:
        if _time_left() < 10:
            break
        
        # Direct optimization
        try:
            h_opt = _optimize_direct(h_init, max_iters=200, maxfun=800)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except:
            pass
        
        if _time_left() < 10:
            break
        
        # G-space optimization
        try:
            g_init = _h_to_g(h_init)
            h_opt_g = _optimize_from_g(g_init, max_iters=200, maxfun=800)
            score_g = _ratio(h_opt_g)
            if score_g > best_score:
                best_score = score_g
                best_h = h_opt_g.copy()
        except:
            pass
    
    # Try DFT spectral optimization with different K values
    for K in [2, 3, 4, 5]:
        if _time_left() < 15:
            break
        
        try:
            h_dft = _dft_spectral_optimize(K=K, max_iters=150, maxfun=600)
            score_dft = _ratio(h_dft)
            if score_dft > best_score:
                best_score = score_dft
                best_h = h_dft.copy()
        except:
            pass
    
    # Final polish: run a high-precision optimization from the best found
    if _time_left() > 20:
        try:
            g_best = _h_to_g(best_h)
            h_polished = _optimize_from_g(g_best, max_iters=500, maxfun=2000)
            score_polished = _ratio(h_polished)
            if score_polished > best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except:
            pass
        
        # Also try direct polish
        if _time_left() > 10:
            try:
                h_direct_polished = _optimize_direct(best_h, max_iters=500, maxfun=2000)
                score_direct_polished = _ratio(h_direct_polished)
                if score_direct_polished > best_score:
                    best_score = score_direct_polished
                    best_h = h_direct_polished.copy()
            except:
                pass
    
    # Scale down to maximize raw score
    # score(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # For h = c * shape: score = shape_score / c
    # So use very small c to maximize raw score
    scale = 1e-8
    result = best_h * scale
    result = np.maximum(result, 0.0)
    
    return list(result)

if __name__ == "__main__":
    result = construct()
    print("Result length:", len(result))
    print("Score:", _ratio(np.array(result)))
