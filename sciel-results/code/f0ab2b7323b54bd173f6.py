"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize, size

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    fsize, size = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

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
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n, beta=5.0):
    """Generate a sinc-like sequence."""
    i = np.arange(n)
    x = (i - n/2) / (n * beta)
    h = np.sinc(x)
    return h

def _gen_gaussian(n, sigma_frac=0.25):
    """Generate a Gaussian window."""
    i = np.arange(n)
    mu = n / 2
    sigma = n * sigma_frac
    return np.exp(-(i - mu)**2 / (2 * sigma**2))

def _gen_barker_like(n):
    """Generate a sequence with alternating pattern."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 in [0, 2]:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _gen_binary_signs(n, seed=0):
    """Generate random binary sign vector."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n)

def _score_binary(s, scale=1.0):
    """Score for binary sign vector s scaled by scale."""
    h = s * scale
    return _score(h)

def _soft_max_abs_ac(ac, beta=50.0):
    """Return soft max abs and the weights for gradient computation."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.ones_like(abs_ac) / len(abs_ac)
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    sm = m + np.log(sum_exp) / beta
    weights = exp_scaled / sum_exp
    return sm, weights

def _binary_gradient_descent(s_init, beta=30.0, max_iters=200, deadline=None, lr=0.1):
    """
    Discrete gradient descent on binary sign vector.
    For each sign s[i], compute derivative of soft_max_abs(autoconv(s)) / (sum(s))^2
    with respect to s[i], then flip if derivative suggests improvement.
    """
    s = s_init.copy()
    n = len(s)
    fsize, size = _get_fft_size(n)
    
    best_val = _score(s)
    best_s = s.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        H = np.fft.rfft(s, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size]
        
        s_sum = np.sum(s)
        if abs(s_sum) < 1e-9:
            break
        
        # Soft max abs and weights
        sm, weights = _soft_max_abs_ac(ac, beta)
        
        # Gradient of soft_max_abs with respect to ac[k]:
        # d(sm)/d(ac[k]) = beta * weights[k] * sign(ac[k])
        grad_ac = beta * weights * np.sign(ac)
        
        # Autoconvolution: ac[k] = sum_i s[i] * s[i-k] (for k >= 0)
        # More precisely, ac[k] = sum_j s[j] * s[j+k] for k in [-n+1, n-1]
        # For the full linear autoconvolution of length 2n-1:
        # ac[m] = sum_j s[j] * s[j + (m - (n-1))] where m in [0, 2n-2]
        # Let's use: ac[m] = sum_j s[j] * s[j + m - (n-1)]
        
        # Gradient of ac[m] with respect to s[i]:
        # d(ac[m])/ds[i] = s[i + m - (n-1)] + s[i - m + (n-1)] (if indices valid)
        # Actually for autoconvolution: ac[m] = sum_j s[j]*s[j+m-(n-1)]
        # d(ac[m])/ds[i] = s[i+m-(n-1)] (if i+m-(n-1) in [0,n-1]) 
        #                  + s[i-m+(n-1)] (if i-m+(n-1) in [0,n-1] and m != n-1)
        # Wait, let me be more careful.
        
        # ac[m] = sum_{j: 0<=j<n, 0<=j+m-(n-1)<n} s[j] * s[j+m-(n-1)]
        # Let k = m - (n-1), so ac[m] = sum_j s[j]*s[j+k] for valid j
        # d(ac[m])/ds[i] = s[i+k] (if 0<=i+k<n) + s[i-k] (if 0<=i-k<n and k!=0)
        #                  (the second term accounts for when i is the second index)
        
        # Gradient of objective with respect to s[i]:
        # f = sm / s_sum^2
        # df/ds[i] = (d(sm)/ds[i] * s_sum^2 - sm * 2 * s_sum) / s_sum^4
        #          = (d(sm)/ds[i]) / s_sum^2 - 2*sm / s_sum^3
        
        # d(sm)/ds[i] = sum_m grad_ac[m] * d(ac[m])/ds[i]
        
        # Compute d(sm)/ds[i] for all i efficiently
        # d(sm)/ds[i] = sum_m grad_ac[m] * (s[i+k_m] + s[i-k_m]) where k_m = m-(n-1)
        
        # This is a convolution-like operation
        # Let's compute it directly
        dsm_ds = np.zeros(n)
        for m in range(size):
            k = m - (n - 1)
            g = grad_ac[m]
            if abs(g) < 1e-15:
                continue
            # Contribution from s[i+k] term: for each i where 0<=i+k<n
            if k >= 0:
                # i + k in [0, n-1] => i in [0, n-1-k]
                dsm_ds[:n-k] += g * s[k:]
            else:
                # i + k in [0, n-1] => i in [-k, n-1]
                dsm_ds[-k:] += g * s[:n+k]
            # Contribution from s[i-k] term (only if k != 0)
            if k != 0:
                if k > 0:
                    # i - k in [0, n-1] => i in [k, n-1]
                    dsm_ds[k:] += g * s[:n-k]
                else:
                    # i - k in [0, n-1] => i in [0, n-1+k]
                    dsm_ds[:n+k] += g * s[-k:]
        
        s_sum_sq = s_sum * s_sum
        s_sum_cu = s_sum_sq * s_sum
        
        # df/ds[i] = dsm_ds[i] / s_sum_sq - 2*sm / s_sum_cu
        grad_f = dsm_ds / s_sum_sq - 2.0 * sm / s_sum_cu
        
        # For binary variables, we flip s[i] if grad_f[i] * s[i] < 0
        # (i.e., moving in the negative gradient direction)
        flip_mask = grad_f * s < 0
        
        if not np.any(flip_mask):
            break
        
        # Flip all signs that improve (or a subset for stability)
        # To avoid oscillation, flip only the most negative ones
        n_flips = max(1, int(0.1 * n))
        flip_indices = np.where(flip_mask)[0]
        if len(flip_indices) > n_flips:
            # Pick the most negative grad_f * s
            scores = grad_f[flip_indices] * s[flip_indices]
            order = np.argsort(scores)[:n_flips]
            flip_indices = flip_indices[order]
        
        s[flip_indices] = -s[flip_indices]
        
        new_val = _score(s)
        if new_val < best_val:
            best_val = new_val
            best_s = s.copy()
        else:
            # Revert
            s[flip_indices] = -s[flip_indices]
            break
    
    return best_s, best_val

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        return h, 1e10
    best_val = _score(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        ac = _autoconv_fft(h)
        s_sq = s * s
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            new_s = s - 2.0 * h[i]
            if abs(new_s) < 1e-9:
                continue
            new_s_sq = new_s * new_s
            
            h[i] = -h[i]
            new_val = _autoconv_max_abs(h) / new_s_sq
            if new_val < best_val:
                best_val = new_val
                s = new_s
                improved = True
                ac = _autoconv_fft(h)
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def _local_search_perturb(h, max_passes=3, deadline=None, step_size=0.1):
    """
    Local optimization with small perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(123)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = old_val + step_size * rng.standard_normal()
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _discrete_polish(h, deadline=None):
    """
    Final polish: try quantizing each element to ±1 and see if it improves.
    """
    h = h.copy()
    best_val = _score(h)
    
    if deadline is not None and time.time() > deadline:
        return h, best_val
    
    # Try rounding to ±1
    h_round = np.sign(h)
    h_round[h_round == 0] = 1.0
    s_round = np.sum(h_round)
    if abs(s_round) > 1e-9:
        val_round = _score(h_round)
        if val_round < best_val:
            best_val = val_round
            h = h_round.copy()
    
    if deadline is not None and time.time() > deadline:
        return h, best_val
    
    # Try local perturbations around best
    rng = np.random.default_rng(999)
    for _ in range(50):
        if deadline is not None and time.time() > deadline:
            break
        h_trial = h.copy()
        n_perturb = rng.integers(5, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            h_trial[idx] += 0.1 * rng.standard_normal()
        s_trial = np.sum(h_trial)
        if abs(s_trial) < 1e-9:
            continue
        val_trial = _score(h_trial)
        if val_trial < best_val:
            best_val = val_trial
            h = h_trial.copy()
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 135.0

    best_score = np.inf
    best_h = np.ones(N) / N

    base_score = _score(best_h)
    best_score = base_score
    best_h = best_h.copy()

    x = np.arange(N)
    
    # Phase 1: Binary sign optimization with discrete gradient descent
    # Try multiple random binary initializations
    rng_main = np.random.default_rng(42)
    
    n_binary_tries = 15
    for trial in range(n_binary_tries):
        if time.time() > deadline - 40:
            break
        
        # Generate diverse binary initializations
        if trial == 0:
            s_init = np.ones(N)
        elif trial == 1:
            s_init = _gen_barker_like(N)
        elif trial == 2:
            # Chirp-based signs
            chirp = _gen_chirp(N)
            s_init = np.sign(chirp)
            s_init[s_init == 0] = 1.0
        elif trial == 3:
            # Cosine-based signs
            s_init = np.sign(np.cos(2 * np.pi * x / N))
            s_init[s_init == 0] = 1.0
        else:
            s_init = _gen_binary_signs(N, seed=trial * 7 + 1)
        
        # Run binary gradient descent
        s_opt, val_opt = _binary_gradient_descent(
            s_init, beta=30.0, max_iters=150, deadline=deadline - 35
        )
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = s_opt.copy()
    
    # Phase 2: Smooth continuous optimization from best binary solution
    if time.time() < deadline - 25:
        h_norm = best_h / np.sum(best_h) if abs(np.sum(best_h)) > 1e-6 else best_h.copy()
        
        for beta in [10, 20, 40, 80]:
            if time.time() > deadline - 20:
                break
            
            def obj(x):
                return _objective_smooth(x, beta=beta)
            
            try:
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 80, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
                        h_norm = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 15:
        h_refined, val_refined = _local_search_flip(best_h, max_passes=4, deadline=deadline-8)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    if time.time() < deadline - 8:
        h_pert, val_pert = _local_search_perturb(best_h, max_passes=2, deadline=deadline-4, step_size=0.05)
        if val_pert < best_score:
            best_score = val_pert
            best_h = h_pert.copy()
    
    # Phase 4: Discrete polish
    if time.time() < deadline - 3:
        h_polished, val_polished = _discrete_polish(best_h, deadline=deadline-1)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
