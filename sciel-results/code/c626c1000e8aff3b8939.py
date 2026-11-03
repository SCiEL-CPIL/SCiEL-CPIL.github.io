"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    """Get the next power of 2 size for FFT of linear convolution of length n."""
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

_FFT_SIZE = _get_fft_size(N)

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    return ac[:2*N-1]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _grad_smooth(h, beta=100.0):
    """
    Analytical gradient of the smooth objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.zeros(N)
    
    ac = _autoconv_fft(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return np.zeros(N)
    
    # Gradient of soft max abs
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    weights = np.exp(scaled)
    weights = weights / np.sum(weights)
    
    # d|ac_k|/dh_i = ac_{k-i} + ac_{k+i} (for valid indices)
    # More precisely, autoconvolution: ac[k] = sum_j h[j]*h[k-j]
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k] (when indices are valid)
    
    # Gradient of soft max: sum_k weights[k] * sign(ac[k]) * d(ac[k])/d(h[i])
    signs = np.sign(ac)
    grad_ac = np.zeros(N)
    
    # For each i, grad_ac[i] = sum_k weights[k] * signs[k] * (h[k-i] + h[i-k])
    # This is a convolution-like operation
    # We can compute this efficiently
    
    # Create a kernel from weights * signs
    kernel = weights * signs
    # The gradient is the cross-correlation of h with kernel, plus the reverse
    # Actually: d(ac[k])/d(h[i]) = h[k-i] + h[i-k] if |k-i| < N, else 0
    # So grad_ac[i] = sum_k kernel[k] * (h[k-i] + h[i-k])
    # This is (kernel * h)[i] + (kernel * h)[i] with appropriate indexing
    
    # Let's compute this via FFT for efficiency
    K = np.fft.rfft(kernel, _FFT_SIZE)
    H = np.fft.rfft(h, _FFT_SIZE)
    # cross-correlation: ifft(conj(K) * H)
    cross = np.fft.irfft(K * H, _FFT_SIZE)
    grad_ac = cross[:N] + cross[N-1:N-1+N][::-1]
    
    # Wait, let me think more carefully.
    # ac[k] = sum_j h[j] h[k-j]
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k] (when 0 <= k-i < N and 0 <= i-k < N)
    
    # grad_obj[i] = sum_k w[k]*sign(ac[k]) * (h[k-i] + h[i-k])
    
    # First term: sum_k w[k]*sign(ac[k]) * h[k-i]
    # Let g[k] = w[k]*sign(ac[k])
    # This is (g * h)[i] where * is cross-correlation
    # cross_corr[i] = sum_k g[k] * h[k-i]
    
    # Using FFT: ifft(conj(FFT(g)) * FFT(h))
    G = np.fft.rfft(g, _FFT_SIZE)
    H = np.fft.rfft(h, _FFT_SIZE)
    cross1 = np.fft.irfft(np.conj(G) * H, _FFT_SIZE)[:N]
    
    # Second term: sum_k g[k] * h[i-k]
    # This is (g * h)[i] where * is convolution
    # conv[i] = sum_k g[k] * h[i-k]
    cross2 = np.fft.irfft(G * H, _FFT_SIZE)[:N]
    
    grad_ac = cross1 + cross2
    
    # Now the full gradient:
    # f = sm / s^2
    # df/dh[i] = (1/s^2) * grad_ac[i] - 2*sm/s^3 * 1
    #          = grad_ac[i]/s^2 - 2*sm/(s^3)
    
    sm = m + np.log(np.sum(weights)) / beta
    grad = grad_ac / (s * s) - 2 * sm / (s ** 3)
    
    return grad

def _gen_gaussian(n, sigma_ratio=0.25):
    """Generate a Gaussian sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n * sigma_ratio
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """Generate a pseudo-random Barker-like sequence."""
    h = np.ones(n)
    for i in range(1, n):
        h[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(noise, kernel, mode='same')
    return smooth

def _local_search_fast(h, max_passes=8, deadline=None):
    """
    Fast local search using incremental autoconvolution updates.
    When flipping h[i], we can update ac efficiently.
    """
    h = h.copy()
    s = np.sum(h)
    ac = _autoconv_fft(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    rng = np.random.default_rng(42)
    
    step_sizes = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # Try flipping signs with incremental update
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_hi = h[i]
            new_hi = -old_hi
            delta = new_hi - old_hi  # = -2*old_hi
            
            # Update autoconvolution: ac_new[k] = ac_old[k] + delta * (h[k-i] + h[i-k])
            # But we need to be careful about boundaries
            
            # Compute the change in ac
            # ac[k] = sum_j h[j] h[k-j]
            # d(ac[k])/d(h[i]) = h[k-i] + h[i-k]
            
            change = np.zeros(2*N-1)
            # For k where 0 <= k-i < N: change[k] += delta * h[k-i]
            valid_k1 = np.arange(i, min(i+N, 2*N-1))
            if len(valid_k1) > 0:
                j1 = valid_k1 - i
                change[valid_k1] += delta * h[j1]
            
            # For k where 0 <= i-k < N: change[k] += delta * h[i-k]
            valid_k2 = np.arange(max(0, i-N+1), min(i+1, 2*N-1))
            if len(valid_k2) > 0:
                j2 = i - valid_k2
                change[valid_k2] += delta * h[j2]
            
            ac_new = ac + change
            s_new = s + delta
            if abs(s_new) < 1e-9:
                continue
            
            max_ac_new = np.max(np.abs(ac_new))
            new_val = max_ac_new / (s_new * s_new)
            
            if new_val < best_val:
                # Accept
                h[i] = new_hi
                s = s_new
                ac = ac_new
                best_val = new_val
                improved = True
            # else: don't change h[i], ac, or s
        
        # Try small perturbations
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            orig = h[i]
            for factor in [1 + step, 1 - step]:
                new_hi = orig * factor
                delta = new_hi - orig
                
                change = np.zeros(2*N-1)
                valid_k1 = np.arange(i, min(i+N, 2*N-1))
                if len(valid_k1) > 0:
                    j1 = valid_k1 - i
                    change[valid_k1] += delta * h[j1]
                
                valid_k2 = np.arange(max(0, i-N+1), min(i+1, 2*N-1))
                if len(valid_k2) > 0:
                    j2 = i - valid_k2
                    change[valid_k2] += delta * h[j2]
                
                ac_new = ac + change
                s_new = s + delta
                if abs(s_new) < 1e-9:
                    continue
                
                max_ac_new = np.max(np.abs(ac_new))
                new_val = max_ac_new / (s_new * s_new)
                
                if new_val < best_val:
                    h[i] = new_hi
                    s = s_new
                    ac = ac_new
                    best_val = new_val
                    improved = True
                    break
            # else: revert (h[i] unchanged)
        
        if not improved:
            break
            
    return h, best_val

def _sa_refine_fast(h, deadline, n_iter=30000):
    """
    Simulated annealing with incremental autoconvolution updates.
    """
    h = h.copy()
    s = np.sum(h)
    ac = _autoconv_fft(h)
    current_val = np.max(np.abs(ac)) / (s * s)
    best_val = current_val
    best_h = h.copy()
    best_s = s
    best_ac = ac.copy()
    rng = np.random.default_rng(123)
    
    T0 = max(current_val * 0.01, 1e-4)
    T_final = max(current_val * 0.001, 1e-6)
    cooling_rate = (T_final / T0) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        i = rng.integers(0, N)
        old_hi = h[i]
        new_hi = -old_hi
        delta = new_hi - old_hi
        
        change = np.zeros(2*N-1)
        valid_k1 = np.arange(i, min(i+N, 2*N-1))
        if len(valid_k1) > 0:
            j1 = valid_k1 - i
            change[valid_k1] += delta * h[j1]
        
        valid_k2 = np.arange(max(0, i-N+1), min(i+1, 2*N-1))
        if len(valid_k2) > 0:
            j2 = i - valid_k2
            change[valid_k2] += delta * h[j2]
        
        ac_new = ac + change
        s_new = s + delta
        if abs(s_new) < 1e-9:
            continue
        
        max_ac_new = np.max(np.abs(ac_new))
        new_val = max_ac_new / (s_new * s_new)
        
        if new_val < current_val:
            # Accept
            h[i] = new_hi
            s = s_new
            ac = ac_new
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
                best_s = s
                best_ac = ac.copy()
        else:
            T = T0 * (cooling_rate ** it)
            delta_val = new_val - current_val
            if delta_val > 0 and rng.random() < np.exp(-delta_val / T):
                h[i] = new_hi
                s = s_new
                ac = ac_new
                current_val = new_val
            # else: revert (h[i] unchanged)
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 148.0  # Tighter safety margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Initialize with diverse candidates
    inits = []
    
    inits.append(np.ones(N))
    inits.append(_gen_gaussian(N))
    inits.append(_gen_chirp(N))
    inits.append(_gen_barker_like(N))
    inits.append(_gen_random_smoothed(N, seed=1))
    inits.append(_gen_random_smoothed(N, seed=2))
    inits.append(_gen_random_smoothed(N, seed=3))
    
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    
    gauss_env = _gen_gaussian(N, sigma_ratio=0.3)
    alt_signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    inits.append(gauss_env * alt_signs)
    
    # Add more diverse initializations
    inits.append(_gen_gaussian(N, sigma_ratio=0.15))
    inits.append(_gen_gaussian(N, sigma_ratio=0.4))
    
    for h_init in inits:
        if time.time() > deadline - 25:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            # Multi-stage gradient optimization with analytical gradients
            betas = [30, 60, 120, 250]
            current_h = h_norm.copy()
            
            for beta in betas:
                if time.time() > deadline - 20:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                def grad(x):
                    return _grad_smooth(x, beta=beta)
                
                res = minimize(obj, current_h, jac=grad, method='L-BFGS-B', 
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-6})
                
                h_opt = res.x
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                
                current_h = h_opt.copy()
                
        except Exception:
            pass

    # Fast local search on the best candidate
    if time.time() < deadline - 12:
        h_refined, val_refined = _local_search_fast(best_h, max_passes=8, deadline=deadline-15)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Simulated annealing refinement
    if time.time() < deadline - 5:
        try:
            h_sa, val_sa = _sa_refine_fast(best_h, deadline=deadline-2, n_iter=30000)
            if val_sa < best_score:
                best_score = val_sa
                best_h = h_sa.copy()
        except Exception:
            pass
    
    # Final gradient polish with high beta
    if time.time() < deadline - 2:
        try:
            s_final = np.sum(best_h)
            if abs(s_final) > 1e-6:
                h_polish_init = best_h / s_final
                
                def obj_polish(x):
                    return _objective_smooth(x, beta=500.0)
                
                def grad_polish(x):
                    return _grad_smooth(x, beta=500.0)
                
                res = minimize(obj_polish, h_polish_init, jac=grad_polish, method='L-BFGS-B', 
                               options={'maxiter': 40, 'disp': False, 'gtol': 1e-7})
                
                h_polish = res.x
                if abs(np.sum(h_polish)) > 1e-6:
                    val_polish = _score(h_polish)
                    if val_polish < best_score:
                        best_score = val_polish
                        best_h = h_polish.copy()
        except Exception:
            pass

    # One more local search pass if time allows
    if time.time() < deadline - 1:
        try:
            h_final, val_final = _local_search_fast(best_h, max_passes=3, deadline=deadline-0.5)
            if val_final < best_score:
                best_score = val_final
                best_h = h_final.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
