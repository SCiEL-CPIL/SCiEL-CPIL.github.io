"""Direct construction: signed step heights minimizing max|autoconv| / (sum)^2."""
import numpy as np
import time

N = 400


def _full_autoconv(f):
    """Compute full autoconvolution via FFT. Returns array of length 2N-1."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    return auto


def _score_from_auto(auto, s):
    """Compute score from autoconvolution array and sum."""
    if abs(s) < 1e-12:
        return float('inf')
    return np.max(np.abs(auto)) / (s * s)


def _update_autoconv_flip(auto, f, i):
    """
    Update autoconvolution after flipping f[i].
    Flipping f[i] from v to -v changes auto[k] by -2*v*f[i+k] for k>=0 and -2*v*f[i-k] for k<0.
    Actually: auto[k] = sum_j f[j]*f[j+k]. Flipping f[i] changes terms where j=i or j+k=i.
    For lag k: auto[k] += -2*f[i]*f[i+k] if i+k in range, and -2*f[i]*f[i-k] if i-k in range.
    But since we're flipping, the change is: new_auto[k] = old_auto[k] - 2*f[i]*(f[i+k] + f[i-k])
    where out-of-range terms are 0.
    """
    n = len(f)
    v = f[i]  # old value
    # Change to autoconv: for each lag k, the contribution from f[i] is removed and negated
    # auto[k] = sum_{j=0}^{n-1-k} f[j]*f[j+k] for k >= 0
    # When we flip f[i], for lag k >= 0:
    #   if 0 <= i <= n-1-k: term f[i]*f[i+k] becomes -f[i]*f[i+k], change = -2*f[i]*f[i+k]
    #   if 0 <= i-k <= n-1-k: i.e., k <= i <= n-1: term f[i-k]*f[i] becomes f[i-k]*(-f[i]), change = -2*f[i]*f[i-k]
    # For lag k < 0: auto[k] = auto[|k|] by symmetry for real signals, but let's handle directly
    # Actually autoconv is defined as (f * f_reversed)[k] or sum_j f[j]*f[j-k]
    # Let's use: auto[k] = sum_j f[j]*f[j+k] for k in -(n-1)..(n-1)
    # Flip f[i]: for lag k, terms involving f[i] are:
    #   j = i: f[i]*f[i+k] valid if 0<=i<n and 0<=i+k<n
    #   j+k = i => j = i-k: f[i-k]*f[i] valid if 0<=i-k<n and 0<=i<n
    # Change in auto[k] = -2*f[i]*f[i+k]*valid1 - 2*f[i]*f[i-k]*valid2
    # But note: if k=0, both terms are f[i]*f[i] = f[i]^2, so change = -2*f[i]*f[i] - 2*f[i]*f[i] = -4*f[i]^2?
    # No wait: when k=0, j=i gives f[i]*f[i], and j+k=i means j=i, same term.
    # So for k=0: only one term f[i]*f[i], change = -2*f[i]*f[i] = -2 (since f[i]^2=1)
    # For k!=0: two distinct terms (unless they coincide, which they don't for k!=0)
    
    new_auto = auto.copy()
    # For k >= 0:
    # Term 1: j=i, need 0 <= i+k < n
    for k in range(0, n):
        if i + k < n:
            new_auto[k] -= 2.0 * v * f[i + k]
        # Term 2: j = i-k, need 0 <= i-k < n, i.e., k <= i
        if k <= i:
            new_auto[k] -= 2.0 * v * f[i - k]
    # For k < 0: auto[k] = auto[-k] for real f? No, autoconv is not necessarily symmetric.
    # auto[k] = sum_j f[j]*f[j+k]
    # For k < 0, let k' = -k > 0: auto[-k'] = sum_j f[j]*f[j-k']
    # Terms involving f[i]: j=i: f[i]*f[i-k'], valid if 0<=i-k'<n
    #               j-k'=i => j=i+k': f[i+k']*f[i], valid if 0<=i+k'<n
    # So for k = -k':
    #   change = -2*v*f[i-k']*(if i-k' in [0,n)) - 2*v*f[i+k']*(if i+k' in [0,n))
    # But note auto[-k'] and auto[k'] are different in general.
    # However, for the autoconvolution as computed by FFT: auto[k] for k<0 corresponds to
    # the other side. Let me just handle it directly.
    for k in range(1, n):
        # k negative: lag = -k
        # auto[-k] = sum_j f[j]*f[j-k]
        # Term 1: j=i: f[i]*f[i-k], valid if 0<=i-k<n
        if i - k >= 0:
            new_auto[-k] -= 2.0 * v * f[i - k]
        # Term 2: j-k=i => j=i+k: f[i+k]*f[i], valid if 0<=i+k<n
        if i + k < n:
            new_auto[-k] -= 2.0 * v * f[i + k]
    
    return new_auto


def _fast_objective(f):
    """Compute max(|autoconv|) / (sum(f))^2 using FFT for full computation."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    auto = _full_autoconv(f)
    return _score_from_auto(auto, s)


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Best tracking
    best_f = np.ones(N)
    best_score = _fast_objective(best_f)
    
    # Phase 1: Try structured initializations
    rng_init = np.random.default_rng(42)
    
    # Random initializations with various densities of -1
    for n_trials in range(20):
        if time.time() > deadline - 100:
            break
        f = rng_init.choice([-1, 1], size=N)
        score = _fast_objective(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Periodic patterns
    for period in range(2, 25):
        if time.time() > deadline - 100:
            break
        for n_flip in range(1, min(period, 12)):
            for trial in range(3):
                if time.time() > deadline - 100:
                    break
                pattern = np.ones(period)
                flip_idx = rng_init.choice(period, size=n_flip, replace=False)
                pattern[flip_idx] = -1
                f = np.tile(pattern, N // period + 1)[:N]
                score = _fast_objective(f)
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
    
    # Phase 2: Fast local search with incremental autoconv updates
    f = best_f.copy()
    auto = _full_autoconv(f)
    s = np.sum(f)
    current_score = _score_from_auto(auto, s)
    
    rng_ls = np.random.default_rng(7)
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline - 20:
        improved = False
        no_improve_count = 0
        order = rng_ls.permutation(N)
        for i in order:
            if time.time() > deadline - 20:
                break
            if no_improve_count > 500:
                break
            
            v = f[i]
            new_auto = _update_autoconv_flip(auto, f, i)
            new_s = s - 2.0 * v
            new_score = _score_from_auto(new_auto, new_s)
            
            if new_score < current_score - 1e-12:
                # Accept the flip
                f[i] = -v
                auto = new_auto
                s = new_s
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    # Phase 3: Multi-restart local search from perturbed best
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Perturb the best solution
        f = best_f.copy()
        n_perturb = max(1, N // 20)
        perturb_idx = rng_ls.choice(N, size=n_perturb, replace=False)
        f[perturb_idx] *= -1
        
        auto = _full_autoconv(f)
        s = np.sum(f)
        current_score = _score_from_auto(auto, s)
        
        # Local search
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline - 5:
            improved = False
            no_improve_count = 0
            order = rng_ls.permutation(N)
            for i in order:
                if time.time() > deadline - 5:
                    break
                if no_improve_count > 300:
                    break
                
                v = f[i]
                new_auto = _update_autoconv_flip(auto, f, i)
                new_s = s - 2.0 * v
                new_score = _score_from_auto(new_auto, new_s)
                
                if new_score < current_score - 1e-12:
                    f[i] = -v
                    auto = new_auto
                    s = new_s
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_f = f.copy()
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
    
    return best_f
