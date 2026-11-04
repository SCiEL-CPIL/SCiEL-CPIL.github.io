"""
Combined program for the second autocorrelation inequality task.
Combines the complex-parameter optimization framework (from the current program, score 1.2196)
with the Sidon-set structural insight (from inspirations, score ~1.1484) to guide initialization
and support selection, while retaining the smooth-max surrogate and L-BFGS-B machinery.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _autoconvolution_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _l2_norm_sq(h):
    """Squared L2 norm."""
    return np.sum(h**2)

def _l1_norm(h):
    """L1 norm."""
    return np.sum(h)

def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.maximum(h, 0.0)
    l2sq = _l2_norm_sq(h)
    l1 = _l1_norm(h)
    peak = _autoconvolution_peak(h)
    if l1 < 1e-30 or peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)

def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-30:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_complex(g_flat, sigma=0.01):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    # Normalize h to have L1 norm 1 to ensure scale invariance and stability
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    h = h / l1
    
    l2sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=sigma)
    
    if peak < 1e-30:
        return 0.0
    
    return -l2sq / peak  # Since l1 is 1, ratio is l2sq / peak

def _optimize_complex(g0, time_limit, sigma=0.01, maxiter=300):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    
    try:
        result = minimize(
            _objective_complex,
            g0,
            args=(sigma,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1500}
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

def _find_sidon_set(n, k, seed=42, deadline=None):
    """Find a Sidon set of size k in {0, ..., n-1} via greedy/random restarts."""
    rng = np.random.RandomState(seed)
    best_set = None
    best_size = 0
    
    for attempt in range(200):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        s = [0]
        sums = {0}
        for _ in range(k - 1):
            found = False
            # Try candidates in random order for diversity
            candidates = list(range(1, n))
            rng.shuffle(candidates)
            for cand in candidates:
                if cand in s:
                    continue
                new_sums = set(sums)
                ok = True
                for idx in s:
                    ns = cand + idx
                    if ns in new_sums:
                        ok = False
                        break
                    new_sums.add(ns)
                if ok:
                    ns = cand + cand
                    if ns in new_sums:
                        ok = False
                    else:
                        new_sums.add(ns)
                if ok:
                    s.append(cand)
                    sums = new_sums
                    found = True
                    break
            if not found:
                break
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            if best_size >= k:
                break
    
    if best_set is None:
        best_set = list(range(min(k, n)))
    return best_set[:k]

def _h_from_support_amplitudes(support, amps):
    """Construct h from support indices and amplitudes."""
    h = np.zeros(N)
    for idx, a in zip(support, amps):
        h[idx] = a
    return h

def _optimize_amplitudes_on_support(support, time_limit, sigma=0.01):
    """
    Optimize amplitudes on a fixed support using L-BFGS-B in g-space.
    We parameterize h = |g|^2 where g is constrained to be zero outside support.
    """
    m = len(support)
    if m < 1:
        return None, 0.0
    
    # Initial amplitudes: equal
    amps = np.ones(m) / np.sqrt(m)
    h0 = _h_from_support_amplitudes(support, amps)
    
    # Build g0: real sqrt(amps) at support indices, zero elsewhere
    g0_real = np.zeros(N)
    g0_imag = np.zeros(N)
    for idx, a in zip(support, amps):
        g0_real[idx] = np.sqrt(max(a, 0.0))
    
    g0_flat = np.concatenate([g0_real, g0_imag])
    
    # Custom objective that enforces support constraint
    def _obj_constrained(g_flat, sigma_val):
        g = g_flat.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        # Zero out non-support entries
        h_masked = np.zeros(N)
        h_masked[support] = h[support]
        
        l1 = np.sum(h_masked)
        if l1 < 1e-30:
            return 0.0
        h_normed = h_masked / l1
        
        l2sq = np.sum(h_normed**2)
        conv = np.convolve(h_normed, h_normed, mode='full')
        peak = _smooth_max(conv, sigma=sigma_val)
        
        if peak < 1e-30:
            return 0.0
        
        return -l2sq / peak
    
    start_time = time.time()
    try:
        result = minimize(
            _obj_constrained,
            g0_flat,
            args=(sigma,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            h_opt = h0
        else:
            g_opt = result.x.reshape(N, 2)
            h_raw = g_opt[:, 0]**2 + g_opt[:, 1]**2
            h_opt = np.zeros(N)
            h_opt[support] = h_raw[support]
            h_opt = np.maximum(h_opt, 0.0)
        
        score = _ratio(h_opt)
        return h_opt, score
    except Exception:
        score = _ratio(h0)
        return h0, score

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N) / N
    best_score = _ratio(best_h)
    
    # Phase 1: Sidon-set guided initialization
    # Try various Sidon set sizes
    for k in [7, 8, 6, 5, 9]:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 40:
            break
        
        sidon = _find_sidon_set(N, k, seed=k * 100, deadline=_DEADLINE)
        if len(sidon) < 3:
            continue
        
        # Optimize amplitudes on this support
        h_opt, s_opt = _optimize_amplitudes_on_support(sidon, time_limit=10.0, sigma=0.01)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Also try with slightly different sigma for the smooth max
        if time.time() < _DEADLINE - 35:
            h_opt2, s_opt2 = _optimize_amplitudes_on_support(sidon, time_limit=8.0, sigma=0.005)
            if h_opt2 is not None and s_opt2 > best_score:
                best_score = s_opt2
                best_h = h_opt2.copy()
    
    # Phase 2: Dense initialization with complex optimization (original strategy)
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 30:
        initializations_h = []
        
        # 1. Uniform
        initializations_h.append(np.ones(N))
        
        # 2. Gaussian-like
        x = np.linspace(-1, 1, N)
        initializations_h.append(np.exp(-x**2 / 0.1))
        initializations_h.append(np.exp(-x**2 / 0.5))
        initializations_h.append(np.exp(-x**2 / 2.0))
        
        # 3. Bimodal
        initializations_h.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
        initializations_h.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
        
        # 4. Triangle / tent
        initializations_h.append(1 - np.abs(x))
        
        # 5. Cosine window
        initializations_h.append(0.5 + 0.5 * np.cos(np.pi * x))
        
        # 6. Random smooth
        for seed in [1, 2, 3, 4, 5]:
            rng_local = np.random.RandomState(seed)
            h_rand = np.abs(rng_local.randn(N))
            kernel = np.ones(5) / 5
            h_smooth = np.convolve(h_rand, kernel, mode='same')
            initializations_h.append(h_smooth)
        
        # 7. Peak in center
        h_center = np.zeros(N)
        h_center[N//2 - 2:N//2 + 3] = 1.0
        initializations_h.append(h_center)
        
        # 8. Peak at edges
        h_edge = np.zeros(N)
        h_edge[0:3] = 1.0
        h_edge[-3:] = 1.0
        initializations_h.append(h_edge)
        
        # 9. Alternating
        h_alt = np.ones(N)
        h_alt[::2] = 0.5
        initializations_h.append(h_alt)
        
        # 10. Linear ramp
        initializations_h.append(np.linspace(0.1, 1.0, N))
        initializations_h.append(np.linspace(1.0, 0.1, N))
        
        # 11. DFT-based
        for seed in [10, 20, 30]:
            rng_local = np.random.RandomState(seed)
            z = rng_local.randn(N) + 1j * rng_local.randn(N)
            H = np.fft.fft(z)
            h_dft = np.abs(H)**2
            h_dft = h_dft / np.max(h_dft)
            initializations_h.append(h_dft)
        
        # 12. More structured patterns
        initializations_h.append(np.sin(np.pi * (x + 1) / 2)**2)
        initializations_h.append(np.cos(np.pi * x / 2)**2)
        
        # 13. Sidon-inspired dense: place small bumps at Sidon positions
        for k in [7, 8]:
            sidon = _find_sidon_set(N, k, seed=k * 200, deadline=_DEADLINE)
            h_sidon_dense = np.zeros(N)
            for idx in sidon:
                # Add a small Gaussian bump at each Sidon position
                h_sidon_dense += np.exp(-((np.arange(N) - idx)**2) / 2.0)
            h_sidon_dense = h_sidon_dense / np.max(h_sidon_dense)
            initializations_h.append(h_sidon_dense)
        
        time_per_opt = 6.0  # seconds per optimization attempt
        
        for i, h0 in enumerate(initializations_h):
            elapsed = time.time() - start_time
            if elapsed + time_per_opt > _DEADLINE - 15:
                break
            
            # Convert h0 to g0
            g0_real = np.sqrt(np.maximum(h0, 0.0))
            g0_imag = np.zeros(N)
            g0_flat = np.concatenate([g0_real, g0_imag])
            
            h_opt = _optimize_complex(g0_flat, time_per_opt, sigma=0.01, maxiter=250)
            
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Phase 3: Final polish with tighter sigma and more iterations
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        # Polish from the best solution with smaller sigma
        g_best_real = np.sqrt(np.maximum(best_h, 0.0))
        g_best_imag = np.zeros(N)
        g_pert_flat = np.concatenate([g_best_real, g_best_imag])
        
        h_polished = _optimize_complex(g_pert_flat, time_limit=8.0, sigma=0.005, maxiter=400)
        score = _ratio(h_polished)
        if score > best_score:
            best_score = score
            best_h = h_polished.copy()
        
        # Try a few perturbations
        for pert_seed in [100, 200, 300, 400, 500]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            g_best_real = np.sqrt(np.maximum(best_h, 0.0))
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.03
            pert_imag = rng_local.randn(N) * 0.03
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            h_opt = _optimize_complex(g_pert_flat, time_limit=5.0, sigma=0.005, maxiter=300)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Ensure nonnegativity and valid output
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
