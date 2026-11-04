"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Combines the Sidon-set spike structural insight (from cell findings) with
complex-parameter L-BFGS-B refinement (from inspiration 1) and a smooth-max
objective for differentiability.

Key structural idea:
- The ratio R = ||h||^2_2 / (||h||_1 * ||h*h||_inf) is scale-invariant.
- For a sum of K non-overlapping unit spikes (each of width 1), 
  ||h||^2_2 = K, ||h||_1 = K, and ||h*h||_inf = 1 (if spikes don't overlap
  in the convolution). So R = K / (K * 1) = 1. That's not helpful directly.
  
  Wait, let me reconsider. If h is a single delta spike of height H at one point:
  ||h||^2_2 = H^2, ||h||_1 = H, (h*h) is a spike of height H^2 at that point.
  R = H^2 / (H * H^2) = 1/H. That goes to 0 as H grows.
  
  Actually the cell finding says "Sidon set of spikes normalized to L1=1" gives
  score proportional to number of non-interacting spikes. Let me re-examine.
  
  If h has K spikes each of height 1/K (so L1 = 1), and they are far enough apart
  that the autoconvolution peaks don't overlap:
  ||h||^2_2 = K * (1/K)^2 = 1/K
  ||h||_1 = 1
  ||h*h||_inf = (1/K)^2 (peak of one spike's self-convolution)
  R = (1/K) / (1 * (1/K)^2) = (1/K) / (1/K^2) = K
  
  So R = K! That's the key. More non-interacting spikes = higher ratio.
  
  On a grid of 50 points in [-1/4, 1/4], spacing = 0.5/49 ≈ 0.0102.
  For spikes to not interact in convolution, they need to be separated by more
  than the spike width. If we use single-point spikes (width 1 grid point),
  two spikes at positions i and j have their self-convolutions at 2i and 2j.
  For non-interaction, we need |2i - 2j| >= 1, i.e., |i-j| >= 1. 
  But adjacent points give 2i and 2i+2, which are 2 apart, so they don't overlap.
  Actually for single-point spikes, h*h at position 2i is h[i]^2, and at 2j is h[j]^2.
  These are at different positions as long as i != j. So ALL 50 points are non-interacting!
  
  Wait, but the convolution is sampled on the full 2N-1 grid. Let me be more careful.
  h has N=50 values. np.convolve(h, h, 'full') has 2N-1 = 99 values.
  If h[i] = a and h[j] = b with i != j, then (h*h)[2i] includes a^2, (h*h)[2j] includes b^2,
  and (h*h)[i+j] includes 2ab. The max of the convolution could be at i+j if 2ab > a^2 and 2ab > b^2.
  
  So for single-point spikes of equal height, the cross-terms at i+j can dominate.
  To avoid this, we need the spikes to be "Sidon" - all pairwise sums distinct.
  A Sidon set in {0,1,...,49} has at most ~sqrt(49) ≈ 7 elements where all pairwise sums are distinct.
  
  Actually, the maximum size of a Sidon set in {0,...,n-1} is about sqrt(n).
  For n=50, max Sidon set size is around 7-8.
  
  But the cell finding measured 1.148, which is only slightly above 1. So maybe the
  improvement is modest. Let me just implement the best strategy:
  
  1. Use complex parameterization h = |g|^2 for natural non-negativity
  2. Use smooth max for differentiability  
  3. Use L-BFGS-B optimization
  4. Include Sidon-set-like sparse initializations
  5. Multiple diverse restarts with time management
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 115.0  # safety margin before 120s kill

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

def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-30:
        return 1e-30
    x_shifted = x - x_max
    return x_max + sigma * np.log(np.sum(np.exp(x_shifted / sigma))) / len(x)

def _objective(g_flat):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    
    l2sq = np.sum(h_norm * h_norm)
    
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h_norm, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    peak = _smooth_max(conv, sigma=0.05)
    
    if peak < 1e-30:
        return 0.0
    
    return -l2sq / peak

def _optimize_from_g(g0, max_iters=300, maxfun=1500):
    """Run L-BFGS-B optimization on complex parameters."""
    result = minimize(
        _objective,
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
    # Greedy construction
    S = []
    for i in range(n):
        candidate = i
        # Check if adding candidate maintains Sidon property
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
    
    # 12. DFT-based
    for seed in [10, 20, 30, 40]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = h_dft / (np.max(h_dft) + 1e-15)
        initializations_h.append(h_dft)
    
    # 13. More structured patterns
    initializations_h.append(np.sin(np.pi * (x + 1) / 2)**2)
    initializations_h.append(np.cos(np.pi * x / 2)**2)
    initializations_h.append(1 + 0.5 * np.sin(2 * np.pi * x))
    initializations_h.append(1 + 0.5 * np.sin(4 * np.pi * x))
    
    # 14. Sparse random (few non-zero entries)
    for seed in [50, 51, 52, 53, 54]:
        rng_local = np.random.RandomState(seed)
        h_sparse = np.zeros(N)
        k = rng_local.randint(3, 10)
        idx = rng_local.choice(N, size=k, replace=False)
        h_sparse[idx] = 1.0
        initializations_h.append(h_sparse)
    
    # Time budget per optimization attempt
    time_per_opt = 6.0
    
    for i, h0 in enumerate(initializations_h):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        # Quick evaluation of the initial point
        score0 = _ratio(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
        
        # Convert to g-space and optimize
        g0_flat = _h_to_g(h0)
        
        try:
            h_opt = _optimize_from_g(g0_flat, max_iters=200, maxfun=1000)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: run additional optimizations from the best solution with perturbations
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        for pert_seed in [100, 200, 300, 400, 500]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            
            g_best_real = np.sqrt(np.maximum(best_h, 0))
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.1
            pert_imag = rng_local.randn(N) * 0.1
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            try:
                h_opt = _optimize_from_g(g_pert_flat, max_iters=150, maxfun=800)
                score = _ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)
