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

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs_and_grad(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp, with gradient w.r.t. ac.
    Returns (value, grad_ac) where grad_ac has same shape as ac.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(ac)
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp < 1e-15:
        return m, np.zeros_like(ac)
    log_sum = np.log(sum_exp)
    value = m + log_sum / beta
    # d/dac_i of max|ac| ≈ (1/beta) * exp(beta*(|ac_i|-m)) / sum_j exp(beta*(|ac_j|-m)) * sign(ac_i)
    weights = exp_scaled / sum_exp / beta
    grad_ac = weights * np.sign(ac)
    return value, grad_ac

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm, _ = _soft_max_abs_and_grad(ac, beta)
    return sm / (s * s)

def _objective_smooth_grad(h, beta=100.0):
    """
    Smooth objective with analytical gradient.
    f(h) = soft_max_abs(ac(h)) / (sum(h))^2
    ac(h) = h * h (linear autoconvolution)
    d ac_k / d h_i = h_{k-i} + h_{k-i} for valid indices (by symmetry, it's 2*h_{k-i} when both valid)
    Actually: ac_k = sum_j h_j * h_{k-j}, so d ac_k / d h_i = h_{k-i} + h_{k-i} = 2*h_{k-i} if 0<=k-i<n, else h_{k-i} if only one term... 
    Wait, ac_k = sum_{j=0}^{n-1} h_j * h_{k-j} where h_m=0 for m<0 or m>=n.
    d ac_k / d h_i = sum_j [delta_{j,i} * h_{k-j} + h_j * delta_{k-j,i}] = h_{k-i} + h_{k-i} = 2*h_{k-i} if 0<=k-i<n
    But careful: if k-i is out of range, the term is 0.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    n = len(h)
    ac = _autoconv_fft(h)
    sm, grad_ac = _soft_max_abs_and_grad(ac, beta)
    # f = sm / s^2
    # df/dh_i = (1/s^2) * sum_k grad_ac_k * d ac_k / d h_i - 2*sm/s^3 * 1
    # d ac_k / d h_i = 2 * h_{k-i} if 0 <= k-i < n, else 0
    # So sum_k grad_ac_k * 2 * h_{k-i} = 2 * (grad_ac * h)_{i} where we use cross-correlation
    # Actually: sum_k grad_ac_k * h_{k-i} is the cross-correlation of grad_ac and h evaluated at i
    # Let's compute: for each i, sum over k: grad_ac[k] * h[k-i] (with h out of range = 0)
    # This is like np.convolve(grad_ac, h, 'full') but we need specific indices
    # sum_k grad_ac[k] * h[k-i] for k-i in [0,n-1] means k in [i, i+n-1]
    # = sum_{j=0}^{n-1} grad_ac[i+j] * h[j] = cross_corr(grad_ac, h, i)
    # We can compute this via FFT: cross_corr = ifft(fft(grad_ac) * conj(fft(h)))
    fsize = _get_fft_size(n)
    G = np.fft.rfft(grad_ac, fsize)
    H = np.fft.rfft(h, fsize)
    # cross_corr[i] = sum_j grad_ac[i+j] * h[j] for j=0..n-1, i=0..n-1
    # This is the "valid" part of cross-correlation
    cc = np.fft.irfft(G * H, fsize)
    # cc[k] = sum_j grad_ac[j] * h[k-j] ... let me think more carefully
    # Actually np.correlate(a, b, 'full')[k] = sum_j a[j] * b[j - k + (len(b)-1)]
    # Let me just use a direct approach via FFT for the sum we need
    # We want: for each i in 0..n-1: sum_{k=0}^{2n-2} grad_ac[k] * (2 * h[k-i] if 0<=k-i<n else 0)
    # = 2 * sum_{j=0}^{n-1} grad_ac[i+j] * h[j]
    # Define v[j] = h[j], w[k] = grad_ac[k]
    # sum_{j=0}^{n-1} w[i+j] * v[j] = cross-correlation of w and v at lag i
    # Using FFT: ifft(fft(w) * conj(fft(v))) gives cross-correlation
    # Actually: cross_corr[i] = sum_j w[i+j] * v[j] = sum_j v[j] * w[i+j]
    # Let's use: ifft(fft(w, fsize) * conj(fft(v, fsize)))
    W = np.fft.rfft(grad_ac, fsize)
    V = np.fft.rfft(h, fsize)
    cross = np.fft.irfft(W * np.conj(V), fsize)
    # cross[k] = sum_j grad_ac[j] * h[j-k] ... need to verify indexing
    # For our purpose: sum_j h[j] * grad_ac[i+j] 
    # Let's compute it directly for small n to verify, or use a different approach
    # Alternative: use np.convolve
    # sum_{j=0}^{n-1} h[j] * grad_ac[i+j] = (h * grad_ac_flipped)[i + (n-1)] ... this is getting complex
    # Let me just compute it via the full convolution approach
    # Actually, let's use: result[i] = sum_{j=0}^{n-1} h[j] * grad_ac[i+j]
    # This is equivalent to: take grad_ac, multiply element-wise with a shifted h
    # For efficiency, let's use FFT-based cross-correlation properly
    # cross_corr[m] = sum_j h[j] * grad_ac[j+m] for m = 0..n-1
    # = ifft(fft(h) * conj(fft(grad_ac))) ... no
    # Standard: (a ⋆ b)[n] = sum_k a[k] b[n+k] = ifft(fft(a) * conj(fft(b)))
    # So (h ⋆ grad_ac)[m] = sum_j h[j] * grad_ac[m+j] = what we want for m=i
    hc = np.fft.irfft(np.fft.rfft(h, fsize) * np.conj(np.fft.rfft(grad_ac, fsize)), fsize)
    # hc[m] = sum_j h[j] * grad_ac[m+j] for valid m
    sum_term = 2.0 * hc[:n]
    
    grad_h = (1.0 / (s * s)) * sum_term - (2.0 * sm / (s * s * s)) * np.ones(n)
    return sm / (s * s), grad_h

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
    # Simple deterministic pseudo-random pattern
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

def _local_search(h, max_passes=5, deadline=None):
    """
    Local search: try flipping signs or small perturbations.
    Uses the exact score function with precomputed FFT for speed.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    n = len(h)
    fsize = _get_fft_size(n)
    
    # Adaptive step sizes for perturbation
    step_sizes = [0.2, 0.1, 0.05, 0.02]
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        step = step_sizes[min(pass_num, len(step_sizes)-1)]
        
        # Precompute FFT for faster updates
        H_fft = np.fft.rfft(h, fsize)
        
        # 1. Try flipping signs
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            # Update FFT incrementally: H_new = H_old - 2*h_old[i]*exp(-2*pi*i*k/fsize)
            # For rfft, we update the full complex FFT then take real part
            delta = 2.0 * h[i]  # h[i] is now -old, so change is -2*old = 2*new... let me be careful
            # old_h = -h[i] (since we flipped), new_h = h[i]
            # change = new_h - old_h = h[i] - (-h[i]) = 2*h[i]
            # But h[i] is the NEW value after flip. old value was -h[i].
            # So delta in h is 2*h[i] (where h[i] is current/new value)
            # Actually this is getting confusing. Let me just recompute.
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
                H_fft = np.fft.rfft(h, fsize)
            else:
                h[i] = -h[i] # revert
        
        # 2. Try small multiplicative perturbations
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            orig = h[i]
            # Try scaling up and down
            for factor in [1 + step, 1 - step]:
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
            else:
                h[i] = orig # revert if no improvement
        
        if not improved:
            break
            
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0 # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Initialize with a few diverse candidates
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random Smoothed
    inits.append(_gen_random_smoothed(N, seed=1))
    inits.append(_gen_random_smoothed(N, seed=2))
    
    # 6. Cosine window
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    
    # 7. Alternating signs with Gaussian envelope
    gauss_env = _gen_gaussian(N, sigma_ratio=0.3)
    alt_signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    inits.append(gauss_env * alt_signs)
    
    # 8. Random binary-like initializations (from inspiration)
    rng = np.random.default_rng(123)
    for seed in [3, 7, 11]:
        rng_local = np.random.default_rng(seed)
        h_rand = rng_local.choice([-1.0, 1.0], size=N)
        # Smooth it slightly
        kernel = np.ones(5) / 5
        h_rand_smooth = np.convolve(h_rand, kernel, mode='same')
        inits.append(h_rand_smooth)

    for h_init in inits:
        if time.time() > deadline - 30:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Gradient Optimization with analytical gradients and increasing beta
        try:
            # Start with lower beta for broader search, then refine
            betas = [50, 100, 200]
            current_h = h_norm.copy()
            
            for beta in betas:
                if time.time() > deadline - 20:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                def obj_grad(x):
                    val, grad = _objective_smooth_grad(x, beta=beta)
                    return val, grad
                
                res = minimize(obj_grad, current_h, method='L-BFGS-B', 
                               options={'maxiter': 150, 'disp': False, 'gtol': 1e-6, 'maxfun': 300})
                
                h_opt = res.x
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                
                # Update current_h for next beta stage
                current_h = h_opt.copy()
                
        except Exception:
            pass

    # Final Local Search on the best candidate found so far
    if time.time() < deadline - 10:
        h_refined, val_refined = _local_search(best_h, max_passes=10, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
            
    # Optional: One more gradient polish on the refined result with high beta
    if time.time() < deadline - 5:
        try:
            s_final = np.sum(best_h)
            if abs(s_final) > 1e-6:
                h_polish_init = best_h / s_final
                
                def obj_polish(x):
                    return _objective_smooth(x, beta=500.0)
                
                def obj_polish_grad(x):
                    val, grad = _objective_smooth_grad(x, beta=500.0)
                    return val, grad
                
                res = minimize(obj_polish_grad, h_polish_init, method='L-BFGS-B', 
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-8, 'maxfun': 500})
                
                h_polish = res.x
                if abs(np.sum(h_polish)) > 1e-6:
                    val_polish = _score(h_polish)
                    if val_polish < best_score:
                        best_score = val_polish
                        best_h = h_polish.copy()
        except Exception:
            pass

    # Ensure we return a valid sequence
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
