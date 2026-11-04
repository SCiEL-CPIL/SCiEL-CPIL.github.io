import numpy as np
import time
from scipy.optimize import minimize

N = 50
dx = 0.5 / N
_DEADLINE = 140.0  # safety margin before 150s kill

def _compute_ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (dx * l1 * max_conv)

def _smooth_max(x, sigma=0.005):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max <= 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_complex(g_flat, sigma=0.005):
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
    h = h / l1
    l2sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=sigma)
    if peak < 1e-15:
        return 0.0
    return -l2sq / (dx * peak)

def _optimize_complex(g0, time_limit, sigma=0.005):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    try:
        result = minimize(
            _objective_complex,
            g0,
            args=(sigma,),
            method='L-BFGS-B',
            options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000}
        )
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            g = g0.reshape(N, 2)
            h = g[:, 0]**2 + g[:, 1]**2
            return h
        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
        return h_opt
    except Exception:
        g = g0.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        return h

def _sidon_set(n, K, rng):
    """
    Find a Sidon set of size K in Z_n using greedy + local search.
    A Sidon set has all pairwise differences distinct (up to sign).
    """
    # Greedy: pick elements that maximize minimum difference separation
    used = set()
    S = []
    candidates = list(range(n))
    rng.shuffle(candidates)
    
    for c in candidates:
        if len(S) >= K:
            break
        # Check if adding c creates duplicate differences
        valid = True
        for s in S:
            d = abs(c - s) % n
            d = min(d, n - d)
            if d == 0:
                valid = False
                break
            # Check if this difference already exists
            for s2 in S:
                if s2 == s:
                    continue
                d2 = abs(s - s2) % n
                d2 = min(d2, n - d2)
                if d2 == d:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(c)
    
    # Local search: try to swap elements to improve
    for _ in range(100):
        if len(S) >= K:
            break
        # Try adding a new element
        added = False
        for c in range(n):
            if c in S:
                continue
            valid = True
            for s in S:
                d = abs(c - s) % n
                d = min(d, n - d)
                if d == 0:
                    valid = False
                    break
                for s2 in S:
                    if s2 == s:
                        continue
                    d2 = abs(s - s2) % n
                    d2 = min(d2, n - d2)
                    if d2 == d:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(c)
                added = True
                break
        if not added:
            # Try swapping
            swapped = False
            for i in range(len(S)):
                for c in range(n):
                    if c in S:
                        continue
                    S_new = S[:i] + S[i+1:] + [c]
                    # Check if S_new is a valid Sidon set
                    valid = True
                    diffs = set()
                    for a in range(len(S_new)):
                        for b in range(a+1, len(S_new)):
                            d = abs(S_new[a] - S_new[b]) % n
                            d = min(d, n - d)
                            if d in diffs or d == 0:
                                valid = False
                                break
                            diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        S = S_new
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                break
    
    return S[:K]

def _h_from_sidon_mask(K, rng):
    """Generate h from a Sidon set mask in frequency domain."""
    S = _sidon_set(N, K, rng)
    mask = np.zeros(N, dtype=complex)
    for s in S:
        mask[s] = 1.0 + 0j
    # Inverse DFT
    h = np.abs(np.fft.ifft(mask))**2
    h = np.real(h)
    h = np.maximum(h, 0)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _two_window_h(w1_center, w1_width, w2_center, w2_width, rng):
    """Generate h as sum of two Gaussian windows with disjoint support."""
    x = np.linspace(-1, 1, N)
    w1 = np.exp(-(x - w1_center)**2 / w1_width**2)
    w2 = np.exp(-(x - w2_center)**2 / w2_width**2)
    # Ensure some separation
    h = w1 + w2
    h = np.maximum(h, 0)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    initializations_h = []
    x = np.linspace(-1, 1, N)
    
    # 1. Sidon set based initializations (key improvement)
    rng_main = np.random.RandomState(42)
    for K in [5, 7, 10, 13, 15]:
        for trial in range(3):
            h_sidon = _h_from_sidon_mask(K, rng_main)
            if np.sum(h_sidon) > 0:
                initializations_h.append(h_sidon)
    
    # 2. Two-window initializations
    for w1c in [-0.3, -0.2, -0.15]:
        for w2c in [0.3, 0.2, 0.15]:
            for w1w in [0.15, 0.2]:
                for w2w in [0.15, 0.2]:
                    h_two = _two_window_h(w1c, w1w, w2c, w2w, rng_main)
                    initializations_h.append(h_two)
    
    # 3. Gaussian-like
    initializations_h.append(np.exp(-x**2 / 0.1))
    initializations_h.append(np.exp(-x**2 / 0.3))
    initializations_h.append(np.exp(-x**2 / 0.5))
    
    # 4. Bimodal
    initializations_h.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
    initializations_h.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
    initializations_h.append(np.exp(-((x - 0.5)**2) / 0.08) + np.exp(-((x + 0.5)**2) / 0.08))
    
    # 5. Triangle / tent
    initializations_h.append(1 - np.abs(x))
    initializations_h.append(np.maximum(0, 1 - 2*np.abs(x)))
    
    # 6. Cosine window
    initializations_h.append(0.5 + 0.5 * np.cos(np.pi * x))
    initializations_h.append(0.5 + 0.5 * np.cos(2 * np.pi * x))
    
    # 7. Random smooth
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        if np.sum(h_smooth) > 0:
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
    h_alt[::2] = 0.3
    initializations_h.append(h_alt)
    
    # 11. Linear ramp
    initializations_h.append(np.linspace(0.1, 1.0, N))
    initializations_h.append(np.linspace(1.0, 0.1, N))
    
    # 12. DFT-based
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        if np.max(h_dft) > 0:
            h_dft = h_dft / np.max(h_dft)
            initializations_h.append(h_dft)
    
    # 13. More structured patterns
    initializations_h.append(np.sin(np.pi * (x + 1) / 2)**2)
    initializations_h.append(np.cos(np.pi * x / 2)**2)
    initializations_h.append(np.sin(np.pi * x)**2)
    
    # 14. Narrow peaks (more peaky)
    for peak_pos in [0.2, 0.3, 0.4]:
        h_peak = np.zeros(N)
        h_peak[N//4 : N//4 + 2] = 1.0
        initializations_h.append(h_peak)
    
    # 15. Very narrow central peak
    h_vnarrow = np.zeros(N)
    h_vnarrow[N//2] = 1.0
    initializations_h.append(h_vnarrow)
    
    # Sort initializations by initial score (best first) to prioritize good starting points
    scored_inits = []
    for h0 in initializations_h:
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 0)
        if np.sum(h0) > 0:
            s = _compute_ratio(h0)
            scored_inits.append((s, h0))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Time budget per optimization
    n_inits = len(scored_inits)
    time_per_opt = 7.0  # seconds per optimization attempt
    
    for i, (init_score, h0) in enumerate(scored_inits):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        # Convert h0 to g0: g0 = sqrt(h0) (real part), imag = 0
        g0_real = np.sqrt(h0)
        g0_imag = np.zeros(N)
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        h_opt = _optimize_complex(g0_flat, time_per_opt, sigma=0.005)
        
        score = _compute_ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Final polish: high-precision optimization from the best solution
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 15:
        g_best_real = np.sqrt(best_h)
        g_best_imag = np.zeros(N)
        g_best_flat = np.concatenate([g_best_real, g_best_imag])
        
        # Very tight tolerance polish
        h_polished = _optimize_complex(g_best_flat, 10.0, sigma=0.001)
        score_polished = _compute_ratio(h_polished)
        if score_polished > best_score:
            best_score = score_polished
            best_h = h_polished
    
    # Additional perturbations around best
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        for pert_seed in [100, 200, 300, 400, 500]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            g_best_real = np.sqrt(best_h)
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.03
            pert_imag = rng_local.randn(N) * 0.03
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            h_opt = _optimize_complex(g_pert_flat, 4.0, sigma=0.005)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        # One more tight polish
        g_best_real = np.sqrt(best_h)
        g_best_imag = np.zeros(N)
        g_best_flat = np.concatenate([g_best_real, g_best_imag])
        h_final = _optimize_complex(g_best_flat, 5.0, sigma=0.001)
        score_final = _compute_ratio(h_final)
        if score_final > best_score:
            best_score = score_final
            best_h = h_final
    
    return [float(x) for x in best_h]
