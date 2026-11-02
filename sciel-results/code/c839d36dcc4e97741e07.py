import numpy as np
import time

N = 50
DELTA = 1.0 / N

def _build_overlap_matrices():
    num_lags = 2 * N - 1
    M = np.zeros((num_lags, N, N))
    for k in range(num_lags):
        for i in range(N):
            for j in range(N):
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                M[k, i, j] = max(0.0, hi - lo)
    return M

_M_matrices = _build_overlap_matrices()
_NUM_LAGS = 2 * N - 1

def _autoconv_sup(h):
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    return np.max(ff_vals)

def _ratio(h):
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, temperature=0.01):
    h = np.maximum(h, 0)
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    max_val = np.max(ff_vals)
    if max_val < 1e-30:
        return 0.0
    shifted = ff_vals - max_val
    log_sum = max_val + temperature * np.log(np.sum(np.exp(shifted / temperature)))
    return log_sum

def _smooth_ratio(h, temperature=0.01):
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, temperature)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _spectral_mask_search(start_time, deadline, best_h, best_score, n_trials=40):
    """
    Model h as |IDFT(binary_mask)|^2 restricted to low-discrepancy spectral masks.
    This leverages the algebraic structure of quadratic forms to naturally minimize
    autoconvolution sidelobes while maintaining high L2 energy concentration.
    """
    for trial in range(n_trials):
        if time.time() - start_time > deadline:
            break
        
        # Generate a binary spectral mask with controlled sparsity
        n_freq = np.random.randint(6, 22)
        # Prefer low-frequency bins for better concentration
        freqs = np.random.choice(N, n_freq, replace=False)
        # Bias towards center frequencies
        freqs = np.sort(freqs)
        
        # Build the mask in frequency domain
        mask = np.zeros(N, dtype=complex)
        for f in freqs:
            mask[f] = 1.0 + 0.5j * np.random.randn()
        
        # Compute the time-domain signal
        h_freq = np.fft.ifft(mask)
        h_candidate = np.abs(h_freq) ** 2
        
        # Normalize and ensure nonnegativity
        h_candidate = np.maximum(h_candidate, 0)
        s = np.sum(h_candidate)
        if s < 1e-15:
            continue
        h_candidate = h_candidate / s
        
        # Polish with local optimization
        h_polished = _local_polish(h_candidate, start_time, deadline)
        sc = _ratio(h_polished)
        if sc > best_score:
            best_score = sc
            best_h = h_polished.copy()
    
    return best_h, best_score

def _local_polish(h, start_time, deadline, max_iter=200):
    """Local polish using gradient-free coordinate descent."""
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    for iteration in range(max_iter):
        if time.time() - start_time > deadline:
            break
        improved = False
        for i in range(N):
            h_old_i = h[i]
            best_val = h_old_i
            best_ratio = _ratio(h)
            
            lo = 0.0
            hi = max(1.0, 3.0 * h_old_i)
            gr = (np.sqrt(5) - 1) / 2
            
            for _ in range(15):
                if time.time() - start_time > deadline:
                    break
                x1 = hi - gr * (hi - lo)
                x2 = lo + gr * (hi - lo)
                
                h[i] = x1
                r1 = _ratio(h)
                h[i] = x2
                r2 = _ratio(h)
                
                if r1 > r2:
                    hi = x2
                else:
                    lo = x1
            
            best_h_i = (lo + hi) / 2
            h[i] = best_h_i
            r_new = _ratio(h)
            
            if r_new > best_ratio + 1e-12:
                improved = True
            else:
                h[i] = h_old_i
        
        if not improved:
            break
    
    s = np.sum(h)
    if s > 1e-15:
        h = h / s
    return h

def _sidon_set_search(start_time, deadline, best_h, best_score, n_trials=30):
    """
    Search for Sidon-type sets that minimize autoconvolution peaks.
    """
    for trial in range(n_trials):
        if time.time() - start_time > deadline:
            break
        
        # Generate a random Sidon-like set
        n_elements = np.random.randint(6, 16)
        used_sums = set()
        seq = []
        
        for candidate in range(N):
            if len(seq) >= n_elements:
                break
            valid = True
            for i in seq:
                if i + candidate in used_sums:
                    valid = False
                    break
            if valid:
                seq.append(candidate)
                used_sums.add(candidate + candidate)
                for i in seq[:-1]:
                    used_sums.add(i + candidate)
        
        if len(seq) < 4:
            continue
        
        h_candidate = np.zeros(N)
        for idx in seq:
            h_candidate[idx] = 1.0
        
        s = np.sum(h_candidate)
        if s < 1e-15:
            continue
        h_candidate = h_candidate / s
        
        h_polished = _local_polish(h_candidate, start_time, deadline)
        sc = _ratio(h_polished)
        if sc > best_score:
            best_score = sc
            best_h = h_polished.copy()
    
    return best_h, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Spectral mask search
    best_h, best_score = _spectral_mask_search(start_time, DEADLINE - 20, best_h, best_score, n_trials=40)
    
    # Phase 2: Sidon set search
    best_h, best_score = _sidon_set_search(start_time, DEADLINE - 10, best_h, best_score, n_trials=30)
    
    # Phase 3: Final polish
    if time.time() - start_time < DEADLINE - 5:
        h_final = _local_polish(best_h.copy(), start_time, DEADLINE, max_iter=500)
        sc = _ratio(h_final)
        if sc > best_score:
            best_score = sc
            best_h = h_final.copy()
    
    return best_h.tolist()
