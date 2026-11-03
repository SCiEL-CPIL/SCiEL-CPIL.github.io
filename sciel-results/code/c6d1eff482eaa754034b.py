"""
Combined program: Minimizing max |autoconv| / (sum h)^2 using smooth surrogate optimization.

Strategy:
1. Reformulate the problem to minimize f(h) = max_k |c_k| / (sum h)^2, where c = h * h.
2. Use a soft-max approximation for the max operator to make the objective differentiable.
3. Derive analytical gradients for fast L-BFGS-B optimization.
4. Use diverse initializations (Kaiser, Chirp, Random Smooth) to explore the landscape.
5. Apply local search refinements (sign flips) to squeeze out further gains.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import i0

N = 400

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
    Returns the soft-max value and the weights.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(ac)
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    log_sum = np.log(sum_exp)
    S = m + log_sum / beta
    w = exp_scaled / sum_exp
    return S, w

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    S, _ = _soft_max_abs(ac, beta)
    return S / (s * s)

def _grad_smooth(h, beta=50.0):
    """
    Analytical gradient of the smooth objective.
    f = S / s^2 where S = soft_max_abs(ac), s = sum(h)
    
    dS/dh_i = 2 * sum_k w_k * sign(ac_k) * h_{k-i}  (indices adjusted for cross-corr)
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.zeros_like(h)
    
    ac = _autoconv_fft(h)
    S, w = _soft_max_abs(ac, beta)
    
    # v_k = w_k * sign(ac_k)
    v = w * np.sign(ac)
    
    # Gradient of S w.r.t h
    # c_k = sum_j h_j h_{k-j}
    # dc_k/dh_i = h_{k-i} + h_{i-k} (with proper boundary handling via zero-padding in FFT)
    # dS/dh_i = sum_k v_k (dc_k/dh_i) = sum_k v_k h_{k-i} + sum_k v_k h_{i-k}
    # Note: ac is symmetric for real inputs, so v is symmetric if ac is symmetric? No, sign(ac) makes it symmetric.
    # Actually, c_k = c_{2N-2-k}. 
    # Let's use FFT cross-correlation.
    # sum_k v_k h_{k-i} is cross-correlation of v and h.
    
    fsize = 1
    while fsize < 2 * (2 * N - 1):
        fsize *= 2
        
    V = np.fft.rfft(v, fsize)
    H = np.fft.rfft(h, fsize)
    
    # Cross-correlation: ifft(conj(V) * H) or ifft(V * conj(H))?
    # corr(x, y)[k] = sum_n x[n] y[n+k]
    # We want sum_k v_k h_{k-i}. Let m = k-i => k = m+i.
    # sum_m v_{m+i} h_m. This is cross-correlation of v and h at lag i.
    # np.correlate(v, h, 'full')
    # FFT method for cross-corr(x, y): ifft(FFT(x) * conj(FFT(y)))
    
    C = np.fft.irfft(V * np.conj(H), fsize)
    
    # The result C is the circular cross-correlation. 
    # We need the linear part.
    # The term sum_k v_k h_{k-i} corresponds to C shifted.
    # Specifically, if v and h are zero-padded to length fsize, 
    # C[i] = sum_n v[n] h[n-i] ? No.
    # Let's stick to the property: d/dh_i (sum_k v_k c_k) = 2 * (v * h)_i (convolution)
    # Because c = h * h. d(h*h)/dh = 2 * h * (symmetric derivative).
    # Actually, d/dh_i (h*h) = h + flip(h).
    # So dS/dh_i = sum_k v_k (h_{k-i} + h_{i-k}) = 2 * sum_k v_k h_{k-i} (if v is symmetric).
    # v is symmetric because ac is symmetric and w depends on |ac|.
    
    # So we need 2 * cross_corr(v, h).
    # Using FFT: ifft(FFT(v) * conj(FFT(h))) gives cross-correlation.
    # Let's verify indices.
    # x = v, y = h.
    # ifft(X * conj(Y))[n] = sum_k x[k] y[k-n] ?
    # Standard correlation: (x * y)[n] = sum_k x[k] y[n-k].
    # Cross-corr: (x \star y)[n] = sum_k x[k] y[k-n].
    # ifft(X * conj(Y)) = x \star y.
    
    # We want sum_k v_k h_{k-i}. This is (v \star h)_i.
    # So C[i] should be the value we want.
    
    # The FFT output C is circular. We need the linear part.
    # v and h are length N. Padded to fsize.
    # The linear cross-correlation is in C[0:N-1] and C[-N+1:0]?
    # Actually, for zero-padded signals, the linear correlation is contained in the full circular correlation.
    # The term h_{k-i} is non-zero only if 0 <= k-i < N and 0 <= k < N.
    
    # Let's just compute it via direct convolution logic or trust the FFT slice.
    # For x, y of length N, zero-padded to M >= 2N-1.
    # ifft(FFT(x) * conj(FFT(y))) gives the circular correlation.
    # The linear correlation c[i] = sum_k x[k] y[k-i] for i in [-(N-1), N-1].
    # In the FFT array of size M, index i corresponds to lag i (if i < M/2) or i-M (if i > M/2).
    # We want i in [0, N-1] for the gradient vector.
    
    # Wait, the gradient index i corresponds to h_i.
    # dS/dh_i = 2 * sum_k v_k h_{k-i}.
    # Let j = k-i. sum_j v_{j+i} h_j.
    # This is cross-correlation of v and h at lag i.
    # (v \star h)_i.
    
    # C = ifft(V * conj(H)).
    # C[i] for i=0..N-1 gives (v \star h)_i.
    
    dS_dh = 2 * C[:N]
    
    # f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s * 1) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (dS_dh * s - 2 * S) / (s ** 3)
    
    return grad

def _init_kaiser(n, alpha=10.0):
    """Kaiser window initialization."""
    x = np.arange(n)
    mid = (n - 1) / 2.0
    r = np.sqrt(1.0 - ((x - mid) / mid) ** 2)
    h = i0(alpha * r) / i0(alpha)
    s = np.sum(h)
    if abs(s) < 1e-9:
        h += 1e-6
        s = np.sum(h)
    return h / s

def _init_chirp(n, freq=1.0):
    """Chirp signal initialization."""
    i = np.arange(n)
    h = np.cos(np.pi * freq * i * i / n)
    s = np.sum(h)
    if abs(s) < 1e-6:
        h += 0.1
        s = np.sum(h)
    return h / s

def _init_random_smooth(rng, n, smooth_len):
    """Random sequence, smoothed."""
    noise = rng.standard_normal(n)
    kernel = np.ones(smooth_len) / smooth_len
    h = np.convolve(noise, kernel, mode='same')
    s = np.sum(h)
    if abs(s) < 1e-6:
        h = h + 0.1
        s = np.sum(h)
    return h / s

def _local_search_flip(h, deadline, max_passes=3):
    """Local search: try flipping signs of individual elements."""
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    best_val = np.inf
    best_h = None
    
    def update_best(h):
        nonlocal best_val, best_h
        s = np.sum(h)
        if abs(s) < 1e-12:
            return
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Initializations
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Kaiser windows
    for alpha in [5.0, 10.0, 15.0, 20.0]:
        inits.append(_init_kaiser(N, alpha))
        
    # 2. Chirps
    for freq in [0.5, 1.0, 2.0, 5.0]:
        inits.append(_init_chirp(N, freq))
        
    # 3. Random Smooth
    for smooth in [5, 10, 20, 50]:
        inits.append(_init_random_smooth(rng, N, smooth))
        
    # 4. Alternating signs on smooth base
    base = _init_kaiser(N, 10.0)
    for shift in [0, 1]:
        h_alt = base.copy()
        h_alt[shift::2] *= -1
        s = np.sum(h_alt)
        if abs(s) > 1e-6:
            inits.append(h_alt / s)

    # Update best with initializations
    for h in inits:
        update_best(h)
        
    # Optimization Loop
    for h_init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        h_norm = h_init / np.sum(h_init)
        
        # Multi-beta annealing
        # Start with low beta (smooth) to find basin, increase beta (sharp) to refine
        for beta in [10, 25, 50, 100]:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                def obj(x, b=beta):
                    return _objective_smooth(x, b)
                
                def grad(x, b=beta):
                    return _grad_smooth(x, b)
                
                res = minimize(obj, h_norm, jac=grad, method='L-BFGS-B',
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-8, 'maxfun': 100})
                
                h_opt = res.x
                s_opt = np.sum(h_opt)
                if abs(s_opt) < 1e-6:
                    continue
                h_opt = h_opt / s_opt
                update_best(h_opt)
                h_norm = h_opt # Chain the result to next beta
            except Exception:
                pass

    # Local Refinement on the best found
    if best_h is not None and time.time() - start_time < DEADLINE - 2:
        h_ref, val_ref = _local_search_flip(best_h, time.time() + (DEADLINE - (time.time() - start_time)) - 1, max_passes=5)
        update_best(h_ref)

    if best_h is None:
        return np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
