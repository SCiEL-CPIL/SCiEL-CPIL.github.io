"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """Smooth objective for gradient-based optimization."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _grad_smooth(h, beta=50.0):
    """
    Analytical gradient of the smooth objective.
    
    Let ac = autoconv(h), S = sum(h), m = max|ac|, sm = soft_max_abs(ac, beta)
    Objective: f = sm / S^2
    
    Gradient: df/dh_i = (1/S^2) * d(sm)/dh_i - 2*sm/S^3 * d(S)/dh_i
             = (1/S^2) * d(sm)/dh_i - 2*sm/S^3
    
    For d(sm)/dh_i:
    sm = m + (1/beta)*log(sum_j exp(beta*|ac_j| - beta*m))
    
    d(sm)/dh_i = sum_j [exp(beta*|ac_j|-beta*m) / sum_k exp(beta*|ac_k|-beta*m)] * beta * sign(ac_j) * d(ac_j)/dh_i
                - (1/beta) * d(m)/dh_i * [sum_j exp(...) * beta - sum_j exp(...)]  # This term vanishes
                = sum_j [w_j * beta * sign(ac_j) * d(ac_j)/dh_i]
    
    where w_j = exp(beta*|ac_j|-beta*m) / sum_k exp(beta*|ac_k|-beta*m)
    
    And d(ac_j)/dh_i: ac_j = sum_k h_k * h_{j-k}
    d(ac_j)/dh_i = h_{j-i} + h_{i-j} (with appropriate bounds)
    """
    n = len(h)
    ac = _autoconv_fft(h)
    size = len(ac)
    
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return np.zeros(n)
    
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    w = exp_scaled / sum_exp  # weights for soft-max
    
    # d(ac_j)/dh_i = h_{j-i} + h_{i-j} for valid indices
    # We need: grad_i = sum_j w_j * beta * sign(ac_j) * (h_{j-i} + h_{i-j})
    
    # This is a convolution-like operation. Let's compute it efficiently.
    # Define g_j = w_j * beta * sign(ac_j)
    g = w * beta * np.sign(ac)
    
    # grad_i = sum_j g_j * h_{j-i} + sum_j g_j * h_{i-j}
    # First term: correlation of g with h
    # Second term: correlation of g with reversed h
    
    # Use FFT for speed
    fsize = 1
    while fsize < 2 * n + size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    G = np.fft.rfft(np.concatenate([g, np.zeros(fsize - size)]), fsize)
    
    # First term: sum_j g_j * h_{j-i} = (g * h)[i] where * is correlation
    # In FFT: ifft(conj(G) * H)
    term1 = np.fft.irfft(np.conj(G) * H, fsize)[:n]
    
    # Second term: sum_j g_j * h_{i-j} = (g * h)[n-1-i] reversed
    # This is the same as term1 but with g reversed
    G_rev = np.fft.rfft(np.concatenate([g[::-1], np.zeros(fsize - size)]), fsize)
    term2 = np.fft.irfft(np.conj(G_rev) * H, fsize)[:n]
    
    d_sm_dh = term1 + term2
    
    S = np.sum(h)
    if abs(S) < 1e-9:
        return np.zeros(n)
    
    sm = _soft_max_abs(ac, beta)
    grad = d_sm_dh / (S * S) - 2 * sm / (S ** 3)
    
    return grad

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_barker_like(n):
    """Generate a sequence with good autocorrelation properties."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 == 0:
            h[i] = 1.0
        elif i % 4 == 1:
            h[i] = -1.0
        elif i % 4 == 2:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _fast_local_search_flip(h, deadline=None, max_passes=10):
    """
    Fast local optimization using incremental FFT updates.
    
    When we flip h[i] -> -h[i], the change in autoconvolution is:
    delta_ac_j = -2*h[i]*h[j-i] - 2*h[i]*h[i-j] + 2*h[i]*h[j-i] + 2*h[i]*h[i-j]
    Actually: ac_new_j = ac_old_j - 4*h[i]*h[j-i] - 4*h[i]*h[i-j] + 4*h[i]*h[j-i] + 4*h[i]*h[i-j]
    Wait, let me think more carefully.
    
    ac_j = sum_k h_k * h_{j-k}
    When h_i -> -h_i:
    ac_new_j = ac_j - h_i*h_{j-i} - h_{i-j}*h_i + (-h_i)*h_{j-i} + (-h_i)*h_{i-j}
             = ac_j - 2*h_i*h_{j-i} - 2*h_i*h_{i-j}
    
    But h_{j-i} and h_{i-j} are only valid if indices are in range.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    
    # Precompute FFT size
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Initial autoconvolution
    ac = _autoconv_fft(h)
    S = np.sum(h)
    best_val = np.max(np.abs(ac)) / (S * S)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Flip h[i]
            old_hi = h[i]
            h[i] = -old_hi
            S -= 2 * old_hi
            
            if abs(S) < 1e-9:
                h[i] = old_hi
                S += 2 * old_hi
                continue
            
            # Update autoconvolution
            # delta_ac_j = -2*old_hi*h[j-i] - 2*old_hi*h[i-j]
            # We need to compute this for all j
            
            # Efficient update using the fact that this is a convolution
            # Create a vector d where d[k] = -2*old_hi*h[k] for valid k
            # Then delta_ac = d convolved with h (but only the parts affected)
            
            # Simpler: just recompute the affected parts
            # ac[j] depends on h[k] for k where j-k is in [0,n-1]
            # When h[i] changes, ac[j] changes for j where i is in [0,n-1] and j-i is in [0,n-1]
            # So j ranges from i to i+n-1, but also from i-n+1 to i
            
            # Let's just do a direct update for the affected range
            # For j from max(0, i-n+1) to min(size-1, i+n-1):
            #   delta_ac[j] = -2*old_hi*(h[j-i] if 0<=j-i<n else 0) - 2*old_hi*(h[i-j] if 0<=i-j<n else 0)
            
            j_min = max(0, i - n + 1)
            j_max = min(size - 1, i + n - 1)
            
            delta = np.zeros(size)
            for j in range(j_min, j_max + 1):
                d = 0.0
                if 0 <= j - i < n:
                    d += -2 * old_hi * h[j - i]
                if 0 <= i - j < n:
                    d += -2 * old_hi * h[i - j]
                delta[j] = d
            
            ac_new = ac + delta
            new_val = np.max(np.abs(ac_new)) / (S * S)
            
            if new_val < best_val:
                best_val = new_val
                ac = ac_new
                improved = True
            else:
                # Revert
                h[i] = old_hi
                S += 2 * old_hi
                ac = ac - delta
        
        if not improved:
            break
    
    return h, best_val

def _gradient_polish(h, deadline=None, max_iter=50):
    """
    Gradient-based polish using the smooth objective and analytical gradient.
    """
    h = h.copy()
    S = np.sum(h)
    if abs(S) < 1e-9:
        return h, _score(h)
    
    # Normalize to sum 1
    h = h / S
    
    best_val = _score(h)
    beta = 50.0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            obj_val = _objective_smooth(h, beta)
            grad = _grad_smooth(h, beta)
            
            # Simple gradient descent with adaptive step size
            step = 0.01
            for _ in range(10):
                h_new = h - step * grad
                S_new = np.sum(h_new)
                if abs(S_new) < 1e-9:
                    step *= 0.5
                    continue
                h_new = h_new / S_new
                val_new = _score(h_new)
                if val_new < best_val:
                    h = h_new
                    best_val = val_new
                    break
                else:
                    step *= 0.5
            
            if step < 1e-8:
                break
        except Exception:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(3):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
    
    # 8. Sparse random with smoothing
    for i in range(3):
        sparse = rng.choice([-1, 1], size=N)
        kernel = np.ones(5)/5
        smooth = np.convolve(sparse, kernel, mode='same')
        inits.append(smooth)
        
    # Optimization loop using smooth surrogate
    time_per_init = 15.0  # Max time per initialization
    for h_init in inits:
        if time.time() > deadline - 30:  # Reserve 30s for final polish
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization
        try:
            # Start with moderate beta, increase for refinement
            for beta in [30, 60]:
                if time.time() > deadline - 25:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                def grad(x):
                    return _grad_smooth(x, beta=beta)
                
                t_iter = time.time()
                res = minimize(obj, h_norm, method='L-BFGS-B', jac=grad,
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-4})
                h_opt = res.x
                
                # Check validity
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Final local search refinement with fast incremental updates
    if time.time() < deadline - 10:
        h_refined, val_refined = _fast_local_search_flip(best_h, deadline=deadline-5, max_passes=8)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Gradient polish
    if time.time() < deadline - 3:
        h_polished, val_polished = _gradient_polish(best_h, deadline=deadline-1, max_iter=30)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
