"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time
import itertools

N = 50
TOTAL_TIME = 110.0  # safety margin before 120s limit (evaluator kills at 150s but we want margin)

def _autoconv_sup(h):
    """Compute supremum of autoconvolution of h (circular, N points)."""
    # Autoconvolution: (h * h)[k] = sum_i h[i] * h[(i+k) % N]
    # Using FFT for efficiency
    H = np.fft.fft(h)
    conv = np.real(np.fft.ifft(H * H))
    return np.max(conv)

def _score(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * sup|autoconv(h)|)"""
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h * h)
    sup = _autoconv_sup(h)
    if sup < 1e-12:
        return 0.0
    return l2_sq / (l1 * sup)

def _gradient_ascent(h, max_iter=200, lr_init=0.05, tol=1e-8):
    """Gradient ascent on log-ratio to optimize amplitudes."""
    h = h.copy()
    h = np.maximum(h, 1e-10)  # keep positive
    
    best_score = _score(h)
    best_h = h.copy()
    
    lr = lr_init
    prev_score = best_score
    
    for it in range(max_iter):
        l1 = np.sum(h)
        l2_sq = np.sum(h * h)
        sup = _autoconv_sup(h)
        
        if l1 < 1e-10 or sup < 1e-10:
            break
        
        score = l2_sq / (l1 * sup)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        if abs(score - prev_score) < tol * max(1.0, abs(prev_score)):
            break
        prev_score = score
        
        # Gradient of log(score) = log(l2_sq) - log(l1) - log(sup)
        # d/dh_i log(l2_sq) = 2*h_i / l2_sq
        # d/dh_i log(l1) = 1 / l1
        # d/dh_i log(sup): need gradient of sup of autoconv
        # sup is max_k conv[k], conv[k] = sum_i h[i]*h[(i+k)%N]
        # d conv[k] / d h_i = h[(i+k)%N] + h[(i-k)%N]
        # For the argmax k*, gradient contribution is h[(i+k*)%N] + h[(i-k*)%N]
        
        H = np.fft.fft(h)
        conv = np.real(np.fft.ifft(H * H))
        k_star = np.argmax(conv)
        
        # Gradient of sup w.r.t. h_i
        # d conv[k_star] / d h_i = h[(i+k_star) % N] + h[(i-k_star) % N]
        idx_plus = (np.arange(N) + k_star) % N
        idx_minus = (np.arange(N) - k_star) % N
        grad_sup = h[idx_plus] + h[idx_minus]
        
        grad_log = (2.0 * h / l2_sq) - (1.0 / l1) - grad_sup / sup
        
        # Update
        h = h + lr * grad_log
        h = np.maximum(h, 1e-10)
        
        # Adaptive learning rate
        if it % 20 == 0:
            lr = max(lr * 0.8, 1e-6)
    
    return best_h, best_score

def _sidon_set(N):
    """Generate a Sidon-like set of indices for initial structure."""
    # Use a quadratic residue-like construction or simple Sidon set
    # For N=50, a simple approach: use indices from a Sidon sequence
    # Sidon set: all pairwise sums are distinct
    # Simple construction: {i^2 mod p} for prime p, but we need subset of [0, N-1]
    
    # Let's try a greedy Sidon set
    sidon = []
    used_sums = set()
    for i in range(N):
        can_add = True
        for j in sidon:
            s = i + j
            if s in used_sums:
                can_add = False
                break
            used_sums.add(s)
        if can_add:
            sidon.append(i)
            if len(sidon) >= 12:  # reasonable size
                break
    
    return sidon

def _random_sidon_like(N, n_points, rng):
    """Generate a random Sidon-like set."""
    attempts = 0
    while attempts < 1000:
        candidates = rng.choice(N, size=n_points, replace=False)
        is_sidon = True
        sums = set()
        for i in range(len(candidates)):
            for j in range(i, len(candidates)):
                s = candidates[i] + candidates[j]
                if s in sums:
                    is_sidon = False
                    break
                sums.add(s)
            if not is_sidon:
                break
        if is_sidon:
            return set(candidates.tolist())
        attempts += 1
    # Fallback: just random
    return set(rng.choice(N, size=n_points, replace=False).tolist())

def _construct_with_structure(active_indices, n_active, rng):
    """Construct a height vector with mass on active indices."""
    h = np.zeros(N)
    # Random positive amplitudes on active indices
    amps = rng.uniform(0.5, 2.0, size=n_active)
    for i, idx in enumerate(active_indices):
        h[idx] = amps[i]
    return h

def _refine_structure(h, rng, time_left):
    """Try to improve the support structure by swapping indices."""
    best_h = h.copy()
    best_score = _score(h)
    
    active = np.where(h > 1e-8)[0]
    if len(active) < 2:
        return best_h, best_score
    
    n_swaps = min(50, int(time_left * 5))  # scale with time
    
    for _ in range(n_swaps):
        if time.time() - _start_time > TOTAL_TIME:
            break
        
        # Try swapping one active index with an inactive one
        i = rng.randint(0, len(active))
        old_idx = active[i]
        
        # Find inactive indices
        inactive = np.where(h < 1e-8)[0]
        if len(inactive) == 0:
            continue
        new_idx = inactive[rng.randint(0, len(inactive))]
        
        h_trial = h.copy()
        h_trial[old_idx] = 0.0
        h_trial[new_idx] = h[old_idx]
        
        s = _score(h_trial)
        if s > best_score:
            best_score = s
            best_h = h_trial.copy()
            # Update active set
            active[i] = new_idx
    
    return best_h, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Phase 1: Try different Sidon-like structures with random amplitudes
    n_structures = 8
    for s in range(n_structures):
        if time.time() - _start_time > TOTAL_TIME:
            break
        
        n_active = rng.randint(8, 16)
        active = _random_sidon_like(N, n_active, rng)
        h = _construct_with_structure(sorted(active), n_active, rng)
        
        # Gradient ascent on amplitudes
        h_opt, score = _gradient_ascent(h, max_iter=100, lr_init=0.03)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Phase 2: Refine structure of best solution
    if time.time() - _start_time < TOTAL_TIME - 10:
        best_h, best_score = _refine_structure(best_h, rng, TOTAL_TIME - (time.time() - _start_time))
        
        # Another round of gradient ascent after structure change
        h_opt, score = _gradient_ascent(best_h, max_iter=150, lr_init=0.02)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Phase 3: Final polish with finer gradient steps
    if time.time() - _start_time < TOTAL_TIME - 5:
        h_opt, score = _gradient_ascent(best_h, max_iter=200, lr_init=0.005, tol=1e-10)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    
    # Normalize to avoid numerical issues (score is scale-invariant)
    if np.sum(best_h) > 0:
        pass  # keep as is, score is invariant to scaling
    
    # Return as list of floats
    return [float(x) for x in best_h]

_start_time = 0.0
