"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(x, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        soft_max = m + np.log(np.sum(np.exp(shifted))) / beta
        return 2.0 * N * soft_max / (s * s)
    else:
        return 2.0 * N * m / (s * s)

def _dct_optimize(h0, deadline, rng, max_iter=200):
    """Optimize using DCT parameterization with spectral amplitude control."""
    # Decompose h0 into baseline + zero-mean part
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    
    # Transform to DCT domain
    # Type II DCT: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k/(2N))
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    # We'll optimize the DCT coefficients directly
    # The constraint is that the resulting h has non-zero sum (handled by baseline)
    
    def _dct_to_h(coeffs):
        """Convert DCT coefficients back to time domain and add baseline."""
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _dct_objective(coeffs):
        """Objective in DCT coefficient space."""
        h = _dct_to_h(coeffs)
        return _soft_max_obj(h)
    
    # Initial coefficients
    coeff_init = dct_coeffs.copy()
    
    # Optimize the DCT coefficients
    try:
        res = minimize(_dct_objective, coeff_init,
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _dct_to_h(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _spectral_shape_optimize(h0, deadline, rng, max_iter=100):
    """Optimize spectral shape by controlling DCT amplitude spectrum."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    # Work with log-amplitudes to ensure positivity and better conditioning
    # But we need to handle signs, so use separate amplitude and sign
    amplitudes = np.abs(dct_coeffs)
    signs = np.sign(dct_coeffs)
    
    # Log-amplitudes (with small epsilon to avoid log(0))
    log_amps = np.log(amplitudes + 1e-10)
    
    def _reconstruct_from_log_amps(log_amps_vec):
        """Reconstruct h from log-amplitudes and fixed signs."""
        amps = np.exp(log_amps_vec)
        coeffs = signs * amps
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _log_amp_objective(log_amps_vec):
        h = _reconstruct_from_log_amps(log_amps_vec)
        return _soft_max_obj(h)
    
    try:
        res = minimize(_log_amp_objective, log_amps.copy(),
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _reconstruct_from_log_amps(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def _low_freq_spectral_optimize(h0, deadline, rng, num_modes=20):
    """Optimize only low-frequency DCT modes for better spectral shape."""
    baseline = np.mean(h0)
    zero_mean = h0 - baseline
    
    dct_coeffs = dct(zero_mean, type=2, norm='ortho')
    
    # Only optimize first num_modes coefficients (low frequencies)
    # Keep high frequencies fixed
    low_coeffs = dct_coeffs[:num_modes].copy()
    high_coeffs = dct_coeffs[num_modes:].copy()
    
    def _reconstruct_low(low_coeffs_vec):
        """Reconstruct with optimized low modes and fixed high modes."""
        coeffs = np.concatenate([low_coeffs_vec, high_coeffs])
        zm = idct(coeffs, type=3, norm='ortho')
        return baseline + zm
    
    def _low_mode_objective(low_coeffs_vec):
        h = _reconstruct_low(low_coeffs_vec)
        return _soft_max_obj(h)
    
    try:
        res = minimize(_low_mode_objective, low_coeffs.copy(),
                      method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < _objective_hard(h0):
            h_new = _reconstruct_low(res.x)
            return h_new
    except Exception:
        pass
    
    return h0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(5):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns (alternating signs with varying period)
    for period in [2, 3, 4, 5]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Polynomial bases
    for deg in [1, 2, 3]:
        x = np.linspace(-1, 1, N)
        h0 = np.zeros(N)
        for d in range(deg + 1):
            h0 += rng.standard_normal() * x**d
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # 6. New: DCT-shaped initializations
    for num_modes in [5, 10, 20]:
        h0 = np.zeros(N)
        coeffs = np.zeros(N)
        for k in range(num_modes):
            coeffs[k] = rng.standard_normal() * (1.0 / (k + 1))
        zm = idct(coeffs, type=3, norm='ortho')
        h0 = 1.0 + zm  # baseline + zero-mean
        initializations.append(h0)
    
    # Optimize each initialization with DCT-based methods
    total_iterations = 50
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        h_curr = h0.copy()
        
        # Try DCT optimization
        if time.time() < deadline - 25.0:
            h_dct = _dct_optimize(h_curr, deadline, rng, max_iter=30)
            val_dct = _objective_hard(h_dct)
            if val_dct < _objective_hard(h_curr):
                h_curr = h_dct
        
        # Try spectral shape optimization
        if time.time() < deadline - 20.0:
            h_spec = _spectral_shape_optimize(h_curr, deadline, rng, max_iter=30)
            val_spec = _objective_hard(h_spec)
            if val_spec < _objective_hard(h_curr):
                h_curr = h_spec
        
        # Try low-frequency spectral optimization
        if time.time() < deadline - 15.0:
            h_low = _low_freq_spectral_optimize(h_curr, deadline, rng, num_modes=15)
            val_low = _objective_hard(h_low)
            if val_low < _objective_hard(h_curr):
                h_curr = h_low
        
        # Update best
        val_curr = _objective_hard(h_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
        
        # Final direct optimization on best candidate
        if time.time() < deadline - 10.0:
            try:
                res = minimize(_soft_max_obj, h_curr,
                              method='L-BFGS-B',
                              args=(50.0,),
                              options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0]:
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
