import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_FFT = 2 * N - 1


def _autoconv(f):
    """Full linear autoconvolution via FFT."""
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)


def _score(f):
    """Raw C3' score: max|ac| / (sum f)^2. Lower is better."""
    ac = _autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)


def _score_vec(f):
    """Vectorized score computation."""
    F = np.fft.rfft(f, n=N_FFT)
    ac = np.fft.irfft(F * F, n=N_FFT)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)


def _soft_max_abs_ac(f, beta=40.0):
    """
    Smooth approximation of max|ac| with gradient.
    Returns (value, grad) where grad is d(soft_max_abs_ac)/df.
    """
    F = np.fft.rfft(f, n=N_FFT)
    ac = np.fft.irfft(F * F, n=N_FFT)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return max_ac, np.zeros(N)

    # log-sum-exp for numerical stability
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted)) / beta + max_ac

    weights = exp_shifted / np.sum(exp_shifted)
    signs = np.sign(ac)

    # grad[i] = 2 * sum_k weights[k] * sign(ac[k]) * f[k-i] for valid k-i
    # This is a cross-correlation. We can compute it via FFT.
    # Let g[k] = weights[k] * signs[k] for k in [0, 2N-2]
    # grad[i] = 2 * sum_{k} g[k] * f[k-i] where 0 <= k-i < N
    # = 2 * sum_{j=0}^{N-1} g[i+j] * f[j] for i in [0, N-1]
    # This is the cross-correlation of g with f, evaluated at lags 0..N-1
    # We can compute via FFT: convolve reversed g with f
    g = weights * signs  # length 2N-1
    # We need: for each i in [0,N-1]: sum_{j=0}^{N-1} g[i+j] * f[j]
    # This is the cross-correlation: (g ⊛ f)[i] where ⊛ denotes cross-correlation
    # Cross-correlation of g and f at lag i = sum_j g[i+j] f[j]
    # = conv(g_reversed, f) at appropriate index
    # Actually: sum_j g[i+j] f[j] = sum_j g[i+j] f[j]
    # Let's use the definition: this is the cross-correlation R_{gf}[i] = sum_j g[i+j] f[j]
    # We can compute this as: ifft(fft(g) * conj(fft(f))) at the right indices
    # But g has length 2N-1 and f has length N, so we need to be careful.
    
    # Simpler: just use the direct formula with vectorized operations
    # grad[i] = 2 * sum_{j=0}^{N-1} g[i+j] * f[j] for i in [0, N-1]
    # We can compute this using a matrix product or sliding dot products
    # For N=400, a loop over i with vectorized inner products is fine
    grad = np.zeros(N)
    for i in range(N):
        # j ranges from 0 to N-1, g[i+j] where i+j ranges from i to i+N-1
        # g has length 2N-1 = 799, so i+N-1 <= N-1+N-1 = 2N-2 which is valid
        grad[i] = 2.0 * np.dot(g[i:i+N], f)
    
    return log_sum_exp, grad


def _objective_and_grad(f):
    """
    Scale-invariant objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    """
    smax, grad_smax = _soft_max_abs_ac(f)
    s = np.sum(f)
    
    if abs(s) < 1e-15 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(abs(s))
    
    return obj, g1 + g2


def _gaussian_smooth(f, sigma):
    """Apply Gaussian smoothing in frequency domain."""
    F = np.fft.rfft(f, n=N_FFT)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0)
    mask = np.exp(-2.0 * np.pi**2 * sigma**2 * freqs**2)
    F_smooth = F * mask
    return np.fft.irfft(F_smooth, n=N_FFT)[:N]


def _greedy_peak_shave(f, max_iters=3000, deadline=None):
    """
    Greedy peak shaving: iteratively flip the sign of the element most
    correlated with the current maximum autoconvolution peak.
    """
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(best)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = ac[peak_idx]
        
        # Determine valid indices i such that both i and peak_idx - i are in [0, N-1]
        if peak_idx < N:
            i_start = max(0, peak_idx - (N - 1))
            i_end = min(N - 1, peak_idx)
        else:
            i_start = max(0, peak_idx - N + 1)
            i_end = min(N - 1, peak_idx)
        
        if i_start > i_end:
            break
        
        valid_i = np.arange(i_start, i_end + 1)
        k_vals = peak_idx - valid_i
        
        products = best[valid_i] * best[k_vals]
        same_sign_mask = (np.sign(products) == np.sign(peak_val)) & (np.abs(products) > 1e-15)
        
        if not np.any(same_sign_mask):
            # Try random flip
            idx = np.random.randint(N)
            candidate = best.copy()
            candidate[idx] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
            continue
        
        candidates_idx = valid_i[same_sign_mask]
        candidates_prod = products[same_sign_mask]
        
        # Try top candidates by |product|
        order = np.argsort(np.abs(candidates_prod))[::-1]
        improved = False
        for trial in range(min(8, len(order))):
            fi = candidates_idx[order[trial]]
            candidate = best.copy()
            candidate[fi] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
                improved = True
                break
        
        if not improved and it % 50 == 0:
            # Stuck: try random perturbation
            n_flips = np.random.randint(1, 4)
            indices = np.random.choice(N, n_flips, replace=False)
            candidate = best.copy()
            candidate[indices] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
    
    return best, best_score


def _local_search(f, deadline, max_iters=40000):
    """Local search with diverse move types."""
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        move_type = np.random.randint(4)
        
        if move_type == 0:
            # Single sign flip guided by peak
            ac = _autoconv(best)
            abs_ac = np.abs(ac)
            peak_idx = np.argmax(abs_ac)
            peak_val = ac[peak_idx]
            
            if peak_idx < N:
                i_start = max(0, peak_idx - (N - 1))
                i_end = min(N - 1, peak_idx)
            else:
                i_start = max(0, peak_idx - N + 1)
                i_end = min(N - 1, peak_idx)
            
            if i_start <= i_end:
                valid_i = np.arange(i_start, i_end + 1)
                k_vals = peak_idx - valid_i
                products = best[valid_i] * best[k_vals]
                same_sign_mask = (np.sign(products) == np.sign(peak_val)) & (np.abs(products) > 1e-15)
                
                if np.any(same_sign_mask):
                    candidates_idx = valid_i[same_sign_mask]
                    candidates_prod = products[same_sign_mask]
                    best_local = np.argmax(np.abs(candidates_prod))
                    flip_idx = candidates_idx[best_local]
                    candidate = best.copy()
                    candidate[flip_idx] *= -1
                    sc = _score(candidate)
                    if sc < best_score:
                        best = candidate
                        best_score = sc
        
        elif move_type == 1:
            # Random single flip
            idx = np.random.randint(N)
            candidate = best.copy()
            candidate[idx] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
        
        elif move_type == 2:
            # Random multi-flip
            n_flips = np.random.randint(2, 6)
            indices = np.random.choice(N, n_flips, replace=False)
            candidate = best.copy()
            candidate[indices] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
        
        else:
            # Block flip: flip a contiguous block
            block_len = np.random.randint(3, 15)
            start = np.random.randint(0, N - block_len + 1)
            candidate = best.copy()
            candidate[start:start + block_len] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
    
    return best, best_score


def _spectral_barker_init(seed=0):
    """
    Barker-like initialization: inverse FFT of phase-only mask.
    Then apply short greedy peak shaving.
    """
    rng = np.random.RandomState(seed)
    n_freqs = N_FFT // 2 + 1
    
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    mags = np.ones(n_freqs)
    mags[0] = 0  # zero DC
    
    F = mags * np.exp(1j * phases)
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    if n_freqs > 1:
        F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1])
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    
    s = np.sum(f)
    if s < 0:
        f = -f
    elif abs(s) < 1e-12:
        f = rng.randn(N)
    
    # Short greedy polish
    f, _ = _greedy_peak_shave(f, max_iters=60)
    
    return f


def _random_barker_init(seed=0):
    """Random ±1 initialization with DC correction."""
    rng = np.random.RandomState(seed)
    f = rng.choice([-1.0, 1.0], size=N)
    s = np.sum(f)
    if s < 0:
        f = -f
    elif abs(s) < 1.0:
        # Adjust to make sum larger
        f[np.argmin(f)] = 1.0 if s >= 0 else -1.0
    return f


def _lowfreq_init(seed=0):
    """Low-frequency dominant initialization."""
    rng = np.random.RandomState(seed)
    n_freqs = N_FFT // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    # Tapered magnitude: strong low freq, weak high freq
    freqs = np.arange(n_freqs)
    mags = np.exp(-freqs / 20.0)
    mags[0] = 0
    
    F = mags * np.exp(1j * phases)
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    if n_freqs > 1:
        F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1])
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    s = np.sum(f)
    if s < 0:
        f = -f
    return f


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    best_f = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # === Phase 1: Diverse initializations ===
    inits = []
    
    # Spectral Barker inits
    for r in range(10):
        if time.time() > deadline - 60.0:
            break
        f0 = _spectral_barker_init(seed=r * 137 + 1)
        sc = _score(f0)
        inits.append((f0, sc))
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Random Barker inits
    for r in range(8):
        if time.time() > deadline - 55.0:
            break
        f0 = _random_barker_init(seed=500 + r * 91)
        sc = _score(f0)
        inits.append((f0, sc))
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Low-frequency inits
    for r in range(6):
        if time.time() > deadline - 50.0:
            break
        f0 = _lowfreq_init(seed=2000 + r * 73)
        sc = _score(f0)
        inits.append((f0, sc))
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Some structured starts
    starts_extra = [
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 3 < 2 else -1.0 for i in range(N)]),
    ]
    for f0 in starts_extra:
        s = np.sum(f0)
        if s < 0:
            f0 = -f0
        sc = _score(f0)
        inits.append((f0.copy(), sc))
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Sort inits by score, keep top few
    inits.sort(key=lambda x: x[1])
    top_inits = [x[0] for x in inits[:6]]
    if best_f is not None and best_f not in top_inits:
        top_inits.insert(0, best_f.copy())
    
    # === Phase 2: Gradient optimization from best starts ===
    for idx, f0 in enumerate(top_inits):
        if time.time() > deadline - 25.0:
            break
        
        try:
            # Multiple beta values for soft max
            for beta in [30.0, 50.0]:
                if time.time() > deadline - 25.0:
                    break
                
                def obj_wrapper(x, b=beta):
                    F = np.fft.rfft(x, n=N_FFT)
                    ac = np.fft.irfft(F * F, n=N_FFT)
                    abs_ac = np.abs(ac)
                    max_ac = np.max(abs_ac)
                    if max_ac < 1e-30:
                        return 1e10
                    shifted = b * (abs_ac - max_ac)
                    exp_shifted = np.exp(shifted)
                    log_sum = np.log(np.sum(exp_shifted)) / b + max_ac
                    s = np.sum(x)
                    if abs(s) < 1e-15:
                        return 1e10
                    return np.log(log_sum) - 2.0 * np.log(abs(s))
                
                def grad_wrapper(x, b=beta):
                    F = np.fft.rfft(x, n=N_FFT)
                    ac = np.fft.irfft(F * F, n=N_FFT)
                    abs_ac = np.abs(ac)
                    max_ac = np.max(abs_ac)
                    if max_ac < 1e-30:
                        return np.ones(N) * 1e-6
                    shifted = b * (abs_ac - max_ac)
                    exp_shifted = np.exp(shifted)
                    log_sum = np.log(np.sum(exp_shifted)) / b + max_ac
                    weights = exp_shifted / np.sum(exp_shifted)
                    signs = np.sign(ac)
                    
                    g = weights * signs
                    grad = np.zeros(N)
                    for i in range(N):
                        grad[i] = 2.0 * np.dot(g[i:i+N], x)
                    
                    s = np.sum(x)
                    if abs(s) < 1e-15:
                        return grad
                    return grad / log_sum - 2.0 * np.ones(N) / s
                
                res = minimize(
                    obj_wrapper,
                    f0,
                    jac=grad_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 15}
                )
                
                candidate = res.x
                sc = _score(candidate)
                if sc < best_score:
                    best_f = candidate.copy()
                    best_score = sc
                    
        except Exception:
            pass
    
    # === Phase 3: Greedy peak shaving polish ===
    if best_f is not None and time.time() < deadline - 15.0:
        polished, polished_score = _greedy_peak_shave(best_f, max_iters=5000, deadline=deadline - 15.0)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # === Phase 4: Local search polish ===
    if best_f is not None and time.time() < deadline - 5.0:
        polished, polished_score = _local_search(best_f, deadline - 5.0, max_iters=50000)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # === Phase 5: Final round of new inits if time permits ===
    if time.time() < deadline - 10.0:
        for r in range(8):
            if time.time() > deadline - 8.0:
                break
            f0 = _spectral_barker_init(seed=5000 + r * 113)
            sc = _score(f0)
            if sc < best_score:
                best_f = f0.copy()
                best_score = sc
                if time.time() < deadline - 5.0:
                    polished, polished_score = _greedy_peak_shave(best_f, max_iters=1500, deadline=deadline - 5.0)
                    if polished_score < best_score:
                        best_f = polished
                        best_score = polished_score
    
    if best_f is None:
        best_f = np.ones(N)
    
    return best_f
