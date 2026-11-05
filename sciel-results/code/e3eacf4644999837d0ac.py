"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _objective_exact(h):
    """Exact objective using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_soft(h, beta):
    """Soft-max surrogate objective."""
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    m = np.max(conv)
    if not np.isfinite(m):
        return 1e10
    exp_vals = np.exp(beta * (conv - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return abs(2.0 * N * sm / (N * N))

def _spectral_flatness_obj(h):
    """
    Objective that combines peak minimization with spectral flatness.
    Spectral flatness encourages energy to be spread across frequencies,
    which reduces peak-to-average ratio in the time domain.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    # Normalize to have sum = N
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    
    # Peak value (soft max for differentiability)
    m = np.max(conv)
    beta = 50.0
    exp_vals = np.exp(beta * (conv - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    
    # Spectral flatness: geometric mean / arithmetic mean of power spectrum
    H = fft(h_norm, n=2*N)
    power = np.abs(H)**2
    # Only use positive frequencies (0 to N-1) to avoid double counting
    power_pos = power[:N]
    # Avoid log(0)
    power_pos = np.maximum(power_pos, 1e-12)
    log_mean = np.mean(np.log(power_pos))
    arith_mean = np.mean(power_pos)
    flatness = np.exp(log_mean) / arith_mean  # 1 = perfectly flat
    
    # Combine: minimize peak, maximize flatness (i.e., minimize -flatness)
    peak_term = abs(2.0 * N * sm / (N * N))
    flatness_term = -flatness * 0.1  # Small weight on flatness
    
    return peak_term + flatness_term

def _combined_objective(h, beta=50.0, flatness_weight=0.05):
    """
    Combined objective: soft-max peak + spectral flatness penalty.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    
    # Soft max peak
    m = np.max(conv)
    if not np.isfinite(m):
        return 1e10
    exp_vals = np.exp(beta * (conv - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    peak_val = abs(2.0 * N * sm / (N * N))
    
    # Spectral flatness
    H = fft(h_norm, n=2*N)
    power = np.abs(H)**2
    power_pos = power[:N]
    power_pos = np.maximum(power_pos, 1e-12)
    log_mean = np.mean(np.log(power_pos))
    arith_mean = np.mean(power_pos)
    flatness = np.exp(log_mean) / arith_mean
    
    return peak_val - flatness_weight * flatness

def _simulated_annealing(h_init, n_iterations=5000, t_start=1.0, t_end=0.01):
    """
    Simulated annealing in the discrete binary space.
    We discretize h to {-1, 1} plus a DC offset to explore Golay-like sequences.
    """
    best_h = h_init.copy()
    best_val = _objective_exact(best_h)
    
    # Create binary base: map h to {-1, 1}
    h_bin = np.sign(h_init)
    h_bin[h_bin == 0] = 1.0
    
    # Add a small DC component to keep sum nonzero
    dc = 0.1
    h_current = h_bin + dc
    current_val = _objective_exact(h_current)
    
    T = t_start
    for i in range(n_iterations):
        # Cool down
        T = t_start * (t_end / t_start) ** (i / n_iterations)
        
        # Pick random index to flip
        idx = np.random.randint(0, N)
        old_sign = h_current[idx] - dc
        h_current[idx] = -old_sign + dc
        
        new_val = _objective_exact(h_current)
        
        if new_val < current_val or np.random.random() < np.exp(-(new_val - current_val) / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h_current.copy()
        else:
            # Revert
            h_current[idx] = old_sign + dc
    
    return best_h, best_val

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses combined spectral flattening + peak minimization + simulated annealing.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Quick parametric families
    # Family 1: Cosine waves
    for m in range(1, 30):
        if time.time() > deadline - 100:
            break
        for a in np.linspace(-0.9, 0.9, 19):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _objective_exact(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Family 2: Binary Golay-like sequences with DC offset
    rng = np.random.RandomState(42)
    for trial in range(50):
        if time.time() > deadline - 100:
            break
        h_bin = rng.choice([-1.0, 1.0], size=N)
        for dc in [0.05, 0.1, 0.2, 0.5]:
            h = h_bin + dc
            v = _objective_exact(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Phase 2: Gradient-based optimization with combined objective
    for beta, fw in [(20.0, 0.05), (50.0, 0.05), (100.0, 0.02), (200.0, 0.01)]:
        if time.time() > deadline - 30:
            break
        
        def obj_combined(h):
            return _combined_objective(h, beta=beta, flatness_weight=fw)
        
        rng2 = np.random.RandomState(100 + int(beta))
        for trial in range(4):
            if time.time() > deadline - 25:
                break
            
            if trial == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.1 * rng2.randn(N)
                h_init -= np.mean(h_init)
                h_init += 1.0
                h_init = h_init * (N / np.sum(h_init))
            
            try:
                result = minimize(
                    obj_combined,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _objective_exact(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 3: Simulated annealing on binary + DC representation
    if time.time() < deadline - 40:
        # Try multiple DC offsets
        for dc in [0.0, 0.05, 0.1, 0.2]:
            if time.time() > deadline - 35:
                break
            
            # Initialize from best_h
            h_bin_init = np.sign(best_h - dc)
            h_bin_init[h_bin_init == 0] = 1.0
            
            h_sa, val_sa = _simulated_annealing(
                h_bin_init + dc,
                n_iterations=3000,
                t_start=0.5,
                t_end=0.01
            )
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa.copy()
    
    # Phase 4: Soft-max polish with high beta
    if time.time() < deadline - 20:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 15:
                break
            
            def obj_polish(h):
                return _objective_soft(h, beta)
            
            rng3 = np.random.RandomState(500 + int(beta))
            for trial in range(3):
                if time.time() > deadline - 12:
                    break
                
                if trial == 0:
                    h_init = best_h.copy()
                else:
                    h_init = best_h + 0.02 * rng3.randn(N)
                    h_init -= np.mean(h_init)
                    h_init += 1.0
                    h_init = h_init * (N / np.sum(h_init))
                
                try:
                    result = minimize(
                        obj_polish,
                        h_init,
                        method='L-BFGS-B',
                        options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-14}
                    )
                    
                    h_opt = result.x
                    if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                        v_opt = _objective_exact(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                except Exception:
                    continue
    
    # Phase 5: Coordinate descent polish
    if time.time() < deadline - 10:
        h = best_h.copy()
        step = 0.05
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            idx = np.argsort(np.abs(h))[::-1]
            for i in idx:
                if time.time() > deadline - 5:
                    break
                old = h[i]
                for delta in [step, -step]:
                    h[i] = old + delta
                    s = np.sum(h)
                    if abs(s) < 1e-10:
                        h[i] = old
                        continue
                    v = _objective_exact(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
            if not improved:
                step *= 0.5
                if step < 0.0001:
                    break
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
