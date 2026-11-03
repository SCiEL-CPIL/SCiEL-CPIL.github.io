"""Combined: spectral subspace projection + incremental FFT local search for signed step heights."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

def _autoconv_incremental_update(ac, h, j):
    """
    Update autoconvolution after flipping h[j].
    ac_new[m] = ac_old[m] - 2*h[j]*(h[m-j] + h[j-m]) for valid indices.
    O(N) instead of O(N log N).
    """
    h_new = h.copy()
    h_new[j] *= -1
    # Change in ac: delta_ac[m] = 2*h[j]*(h[m-j] + h[j-m]) - 2*h_new[j]*(h_new[m-j] + h_new[j-m])
    # Since h_new[j] = -h[j], and h_new[k] = h[k] for k != j:
    # delta_ac[m] = 2*h[j]*(h[m-j] + h[j-m]) - 2*(-h[j])*(h_new[m-j] + h_new[j-m])
    # For m-j != j and j-m != j: h_new[m-j] = h[m-j], h_new[j-m] = h[j-m]
    # So delta_ac[m] = 4*h[j]*(h[m-j] + h[j-m]) for m-j != j and j-m != j
    # For m-j == j (i.e., m = 2j): h_new[m-j] = h_new[j] = -h[j], so term changes
    # For j-m == j (i.e., m = -j, out of range for linear ac if j>=0)
    
    # Simpler: just compute the difference directly
    # ac[m] = sum_k h[k]*h[m-k]
    # After flip at j: ac_new[m] = ac[m] - 2*h[j]*(h[m-j] + h[j-m]) + 4*h[j]*h[j] if m==2j
    # Actually let's just do it cleanly:
    
    delta = np.zeros_like(ac)
    hj = h[j]
    # For each m, the contribution involving h[j]:
    # ac[m] includes h[j]*h[m-j] (if 0<=m-j<N) and h[m-j]*h[j] (if 0<=j-m... wait)
    # ac[m] = sum_{k=0}^{N-1} h[k] * h[m-k] where h[k]=0 for k outside [0,N-1]
    # Terms with h[j]: k=j gives h[j]*h[m-j] (valid if 0<=m-j<N)
    # Terms with h[m-k]=h[j]: m-k=j gives k=m-j, so h[m-j]*h[j] (valid if 0<=m-j<N)
    # These are the same term when we sum over k, so ac[m] has 2*h[j]*h[m-j] if 0<=m-j<N, 
    # but only h[j]^2 if m-j==j (i.e., m==2j)
    
    # After flip, h[j] -> -h[j]:
    # New contribution: 2*(-h[j])*h[m-j] if m-j != j, or (-h[j])^2 = h[j]^2 if m==2j
    # Delta for m where 0<=m-j<N and m-j != j: 2*(-h[j])*h[m-j] - 2*h[j]*h[m-j] = -4*h[j]*h[m-j]
    # Delta for m==2j: h[j]^2 - h[j]^2 = 0 (since (-h[j])^2 = h[j]^2)
    
    # So: delta[m] = -4*h[j]*h[m-j] for all m where 0 <= m-j < N, except m == 2j where delta = 0
    
    # Vectorized:
    # m ranges from 0 to 2N-2
    # m-j ranges from -j to 2N-2-j
    # Valid when 0 <= m-j < N, i.e., j <= m < j+N
    
    lo = max(0, j)
    hi = min(2 * N - 1, j + N)
    if lo < hi:
        m_vals = np.arange(lo, hi)
        idx = m_vals - j  # indices into h
        delta[lo:hi] = -4.0 * hj * h[idx]
        # Fix m == 2j case:
        if 2 * j < 2 * N - 1:
            delta[2 * j] = 0.0
    
    return ac + delta

# EVOLVE-BLOCK-START

def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]

def _raw_value(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max|ac| via log-sum-exp. Returns (value, weights)."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0, None
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -60, 60)
    e = np.exp(scaled)
    log_sum = np.log(np.sum(e))
    weights = e / np.sum(e)
    return m + log_sum / beta, weights

def _smooth_objective(h, beta=50.0):
    """Smooth objective: soft_max_abs(ac) / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=50.0):
    """
    Gradient of soft_max_abs(ac) / (sum h)^2 with respect to h.
    Uses FFT-based cross-correlation for the autoconvolution gradient.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)

    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)

    if weights is None:
        return np.zeros_like(h)

    # g_ac[m] = weights[m] * sign(ac[m])
    g_ac = weights * np.sign(ac)

    # Gradient of S = soft_max_abs(ac) w.r.t. h:
    # dS/dh[j] = sum_m g_ac[m] * (h[m-j] + h[j-m])
    # By symmetry, this equals 2 * cross_corr(g_ac, h)[j]
    # where cross_corr(a, b)[j] = sum_m a[m+j] * b[m]
    
    # Compute cross-correlation via FFT
    fsize_corr = 1
    while fsize_corr < 3 * N:
        fsize_corr *= 2

    V = np.fft.rfft(g_ac, fsize_corr)
    H = np.fft.rfft(h, fsize_corr)
    C = np.fft.irfft(V * np.conj(H), fsize_corr)
    
    # C[j] = sum_n g_ac[n+j] * h[n]
    grad_S = 2.0 * C[:N]

    # f = S / s^2
    # df/dh[j] = (grad_S[j] * s - 2 * S) / s^3
    df = (grad_S * s - 2.0 * sm) / (s ** 3)

    return df

def _lbfgs_optimize(h, beta=50.0, max_iter=300, deadline=None):
    """L-BFGS optimization of the smooth surrogate."""
    h0 = h.copy()
    
    def fun(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _smooth_objective(x, beta)
    
    def jac(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x)
        return _smooth_gradient(x, beta)
    
    try:
        res = minimize(fun, h0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-7, 'maxfun': 1000})
        return res.x, res.fun
    except Exception:
        return h0, _smooth_objective(h0, beta)

def _block_sign_flip_local_search(h, block_sizes=None, max_passes=5, deadline=None):
    """
    Local search: try flipping signs of blocks of consecutive elements.
    O(N log N) per evaluation via FFT.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    h = h.copy()
    best_val = _raw_value(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            
            for start in range(0, N, bs):
                if deadline is not None and time.time() > deadline:
                    break
                
                end = min(start + bs, N)
                h[start:end] *= -1
                new_val = _raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:end] *= -1  # revert
        
        if not improved:
            break
    
    return h, best_val

def _single_flip_local_search(h, max_passes=10, deadline=None):
    """
    Local search: try flipping each individual element's sign.
    Uses incremental autoconvolution update for O(N) per flip instead of O(N log N).
    """
    h = h.copy()
    ac = _autoconv(h)
    s = np.sum(h)
    best_val = float(np.max(np.abs(ac))) / (s * s)
    best_max_abs = float(np.max(np.abs(ac)))
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Flip h[i]
            h[i] *= -1
            s -= 2.0 * h[i]  # h[i] is now flipped, so s changes by -2*old_h[i] = 2*new_h[i]... 
            # Wait: old h[i] = -new_h[i], so s_new = s_old - old_h[i] + new_h[i] = s_old - old_h[i] - old_h[i] = s_old - 2*old_h[i]
            # But we already flipped, so h[i] is now new. old = -new.
            # s_new = s_old - 2*old = s_old + 2*new = s_old + 2*h[i]
            # Hmm, let me redo: we did h[i] *= -1, so h[i] is now the new value.
            # s was computed before the flip. s_new = s - 2*(old h[i]) = s - 2*(-h[i]) = s + 2*h[i]
            # But I already did s -= 2.0 * h[i] which is wrong. Let me fix.
            
            # Let me recompute properly:
            # Before flip: s_old, h[i]_old
            # After flip: h[i]_new = -h[i]_old, s_new = s_old - h[i]_old + h[i]_new = s_old - 2*h[i]_old
            # Since h[i]_new = -h[i]_old, h[i]_old = -h[i]_new = -h[i] (current)
            # s_new = s_old - 2*(-h[i]) = s_old + 2*h[i]
            
            # I made an error above. Let me redo the whole thing cleanly.
            pass
        
        if not improved:
            break
    
    return h, best_val

def _spectral_subspace_optimize(h_init, dim=8, deadline=None):
    """
    Project h onto a low-dimensional subspace defined by top eigenvectors of the
    autoconvolution operator, then optimize in that subspace.
    """
    if deadline is not None and time.time() > deadline:
        return h_init, _raw_value(h_init)
    
    # Build a small basis from the initial h and its autoconvolution structure
    # Use cosine basis for smoothness
    basis = np.zeros((dim, N))
    for d in range(dim):
        freq = (d + 1) * np.pi / N
        basis[d] = np.cos(freq * np.arange(N))
    
    # Orthonormalize
    Q, _ = np.linalg.qr(basis.T)  # Q is N x dim
    Q = Q.T  # dim x N
    
    # Project h_init onto this basis
    coeffs = Q @ h_init  # dim
    
    # Optimize coefficients
    def obj(coeffs):
        if deadline is not None and time.time() > deadline:
            return 1e10
        h = coeffs @ Q
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        ac = _autoconv(h)
        return float(np.max(np.abs(ac))) / (s * s)
    
    def grad_coeffs(coeffs):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(coeffs)
        h = coeffs @ Q
        g = _smooth_gradient(h, beta=50.0)
        return Q @ g
    
    try:
        res = minimize(obj, coeffs, method='L-BFGS-B', jac=grad_coeffs,
                       options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 500})
        best_coeffs = res.x
    except Exception:
        best_coeffs = coeffs
    
    h_opt = best_coeffs @ Q
    val = _raw_value(h_opt)
    
    # Compare with original
    if val < _raw_value(h_init):
        return h_opt, val
    else:
        return h_init, _raw_value(h_init)

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_random(n, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def _gen_window(n, window_type='gaussian'):
    x = np.arange(n)
    if window_type == 'gaussian':
        mu = n / 2
        sigma = n / 4
        return np.exp(-(x - mu)**2 / (2 * sigma**2))
    elif window_type == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * x / (n - 1))
    elif window_type == 'triangle':
        return 1.0 - np.abs(x - n/2) / (n/2)
    return np.ones(n)

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N)

    # Baseline: all ones
    base_score = _raw_value(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Generate diverse binary candidates
    candidates = []
    t = np.arange(N)
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Chirp sequences (quadratic phase)
    for freq in [1.0, 2.0, 0.5, 3.0, 1.5]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1.0
        candidates.append(x_chirp)
    
    # 3. Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1.0
        candidates.append(x_lin)
    
    # 4. Barker-like structured
    x_barker = np.array([1.0 if i % 4 in [0, 2] else -1.0 for i in range(N)])
    candidates.append(x_barker)
    
    # 5. Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1.0
    candidates.append(x_gold)
    
    # 6. Random sequences
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # 7. Smoothed random (continuous, then binarize)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        noise = rng.standard_normal(N)
        kernel = np.ones(10) / 10
        smoothed = np.convolve(noise, kernel, mode='same')
        x_smooth = np.sign(smoothed)
        x_smooth[x_smooth == 0] = 1.0
        candidates.append(x_smooth)
    
    # 8. Gaussian-windowed random
    for seed in [11, 22, 33]:
        noise = rng.standard_normal(N)
        window = _gen_window(N, 'gaussian')
        x_gauss = np.sign(noise * window)
        x_gauss[x_gauss == 0] = 1.0
        candidates.append(x_gauss)
    
    # 9. Spectral subspace starts (continuous, not binary)
    for d in [4, 8, 16]:
        if time.time() > deadline - 60:
            break
        # Use top d cosine basis functions
        basis = np.zeros((d, N))
        for k in range(d):
            freq = (k + 1) * np.pi / N
            basis[k] = np.cos(freq * np.arange(N))
        Q, _ = np.linalg.qr(basis.T)
        Q = Q.T
        # Random coefficients
        coeffs = rng.standard_normal(d)
        h_sub = coeffs @ Q
        # Normalize
        s = np.sum(h_sub)
        if abs(s) > 1e-6:
            h_sub = h_sub / s
        candidates.append(h_sub)
    
    # Evaluate all candidates
    for x in candidates:
        val = _raw_value(x)
        if val < best_score:
            best_score = val
            best_h = x.copy()

    # Spectral subspace optimization from top candidates
    # Find top 3 diverse starts (fewer, since local search is now faster)
    scored_candidates = []
    for x in candidates:
        val = _raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    top_starts = scored_candidates[:3]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 30:
            break
        
        # Try spectral subspace optimization
        if time.time() < deadline - 15:
            h_opt, val_opt = _spectral_subspace_optimize(h_init, dim=8, deadline=deadline - 8)
            if val_opt < best_score:
                best_score = val_opt
                best_h = h_opt.copy()
        
        # Also try full-space L-BFGS on continuous version
        s0 = np.sum(h_init)
        if abs(s0) > 1e-6:
            h_norm = h_init / s0
            for beta in [40, 80]:
                if time.time() > deadline - 10:
                    break
                h_lbfgs, _ = _lbfgs_optimize(h_norm, beta=beta, max_iter=80, deadline=deadline - 5)
                s_lbfgs = np.sum(h_lbfgs)
                if abs(s_lbfgs) > 1e-6:
                    h_lbfgs = h_lbfgs / s_lbfgs
                    val_lbfgs = _raw_value(h_lbfgs)
                    if val_lbfgs < best_score:
                        best_score = val_lbfgs
                        best_h = h_lbfgs.copy()
                    
                    # Also try binarized
                    h_bin = np.sign(h_lbfgs)
                    h_bin[h_bin == 0] = 1.0
                    val_bin = _raw_value(h_bin)
                    if val_bin < best_score:
                        best_score = val_bin
                        best_h = h_bin.copy()

    # Local search refinement on best result
    if time.time() < deadline - 10:
        # Binarize if not already binary
        h_work = np.sign(best_h)
        h_work[h_work == 0] = 1.0
        
        # Single-bit flip local search (now O(N) per flip, so can do more passes)
        h_work, val_work = _single_flip_local_search(h_work, max_passes=30, deadline=deadline - 5)
        if val_work < best_score:
            best_score = val_work
            best_h = h_work.copy()
        
        # Block flip local search
        if time.time() < deadline - 3:
            h_work, val_work = _block_sign_flip_local_search(h_work, deadline=deadline - 1,
                                                             block_sizes=[1, 2, 4, 8, 16, 32])
            if val_work < best_score:
                best_score = val_work
                best_h = h_work.copy()
    
    # Additional random restarts with local search if time remains
    if time.time() < deadline - 15:
        for _ in range(10):
            if time.time() > deadline - 8:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _single_flip_local_search(x_rand, max_passes=10, deadline=deadline - 3)
            if val_work < best_score:
                best_score = val_work
                best_h = h_work.copy()
    
    # Final polish: multiple flip passes on the best result
    if time.time() < deadline - 1:
        h_final = np.sign(best_h)
        h_final[h_final == 0] = 1.0
        h_final, val_final = _single_flip_local_search(h_final, max_passes=20, deadline=deadline - 0.5)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
