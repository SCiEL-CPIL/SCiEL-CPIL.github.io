import numpy as np
import time

N = 400
DEADLINE = 135.0  # safety margin before 150s kill

def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _sine_sign_pattern(freq, phase, N=N):
    """
    Generate a signed step function from zero crossings of a sine wave.
    """
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    pattern = np.sin(2 * np.pi * freq * x + phase)
    pattern[pattern == 0] = 1.0
    return np.sign(pattern)

def _spectral_search(start_time):
    """
    Search over sine frequency and phase parameters to find good initial patterns.
    """
    best_pattern = None
    best_score = np.inf
    
    freqs = np.arange(1, 21)
    phases = np.linspace(0, 2*np.pi, 36, endpoint=False)
    
    for f in freqs:
        for ph in phases:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 50:
                return best_pattern, best_score
            
            pattern = _sine_sign_pattern(f, ph)
            score = _compute_c3_bound(pattern)
            
            if score < best_score:
                best_score = score
                best_pattern = pattern.copy()
    
    return best_pattern, best_score

def _block_flip_move(signs, block_size, rng):
    """
    Flip a random contiguous block of signs.
    """
    new_signs = signs.copy()
    start = rng.randint(0, N - block_size + 1)
    new_signs[start:start + block_size] = -new_signs[start:start + block_size]
    return new_signs

def _simulated_annealing(signs, start_temp=0.5, end_temp=0.01, n_iter=3000, 
                         block_sizes=None, rng_seed=12345, time_limit=None, start_time=None):
    """
    Simulated annealing with block-flip moves.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32]
    
    rng = np.random.RandomState(rng_seed)
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        # Check time limit
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Exponential cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iter - 1, 1))
        
        # Choose a random block size with bias toward smaller blocks
        bs = block_sizes[rng.choice(len(block_sizes), p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])]
        if bs > N:
            bs = N
        
        new_signs = _block_flip_move(current, bs, rng)
        new_score = _compute_c3_bound(new_signs)
        
        delta = new_score - current_score
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / max(temp, 1e-10))):
            current = new_signs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score

def _greedy_flip_optimize(signs, max_rounds=20, time_limit=None, start_time=None):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    for rnd in range(max_rounds):
        improved = False
        for i in range(N):
            # Time check
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    return current, current_score
            
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        if not improved:
            break
    
    return current, current_score

def _admm_continuous_relaxation(signs, n_iters=500, time_limit=None, start_time=None):
    """
    ADMM-style optimization on continuous relaxation.
    We minimize the L1 norm of autoconvolution by treating signs as continuous 
    variables in [-1, 1] and using projected gradient descent with soft-thresholding.
    
    The key insight: the autoconvolution C = f * f is a quadratic form in f.
    Its L1 norm can be minimized by gradient descent on the continuous domain,
    then projecting back to {-1, 1} at the end.
    """
    current = signs.astype(np.float64).copy()
    best = current.copy()
    best_score = _compute_c3_bound(current)
    
    dx = 0.5 / N
    n_fft = 2 * N
    
    # Learning rate schedule
    lr_start = 0.1
    lr_end = 0.001
    
    for i in range(n_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Compute gradient of L1(autoconv) w.r.t. f
        # autoconv = ifft(fft(f)^2)
        # d/d f [L1(autoconv)] = sign(autoconv) * 2 * ifft(fft(autoconv) * conj(fft(f)))
        F = np.fft.rfft(current, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft) * dx
        
        # Gradient: 2 * conv_sign * ifft(fft(conv_sign) * conj(fft(f)))
        conv_sign = np.sign(conv)
        # Handle zeros in conv_sign
        conv_sign[conv_sign == 0] = 1.0
        
        F_conv_sign = np.fft.rfft(conv_sign, n=n_fft)
        grad = np.fft.irfft(F_conv_sign * np.conj(F), n=n_fft) * dx * 2.0
        
        # Learning rate decay
        lr = lr_start * (lr_end / lr_start) ** (i / max(n_iters - 1, 1))
        
        # Gradient step
        current = current - lr * grad
        
        # Project to [-1, 1]
        current = np.clip(current, -1.0, 1.0)
        
        # Evaluate discrete score
        discrete = np.sign(current)
        discrete[discrete == 0] = 1.0
        score = _compute_c3_bound(discrete)
        
        if score < best_score:
            best_score = score
            best = discrete.copy()
    
    return best, best_score

def _pairwise_swap_optimize(signs, max_rounds=10, time_limit=None, start_time=None):
    """
    Try swapping pairs of signs to find improvements.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    rng = np.random.RandomState(999)
    
    for _ in range(max_rounds):
        improved = False
        for _ in range(300):
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    return current, current_score
            
            i, j = rng.choice(N, size=2, replace=False)
            if current[i] == current[j]:
                continue
            current[i], current[j] = current[j], current[i]
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i], current[j] = current[j], current[i]
        if not improved:
            break
    
    return current, current_score

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Phase 1: Spectral search over sine parameters
    best_pattern, best_score = _spectral_search(start_time)
    
    if best_pattern is None:
        best_pattern = np.ones(N)
        best_score = _compute_c3_bound(best_pattern)
    
    # Phase 2: Simulated annealing with block flips
    elapsed = time.time() - start_time
    remaining = DEADLINE - 40 - elapsed
    if remaining > 20:
        optimized, opt_score = _simulated_annealing(
            best_pattern, 
            start_temp=0.5, 
            end_temp=0.01, 
            n_iter=4000,
            time_limit=remaining,
            start_time=start_time
        )
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Phase 3: ADMM continuous relaxation
    elapsed = time.time() - start_time
    remaining = DEADLINE - 30 - elapsed
    if remaining > 15:
        optimized, opt_score = _admm_continuous_relaxation(
            best_pattern,
            n_iters=800,
            time_limit=remaining,
            start_time=start_time
        )
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Phase 4: Greedy coordinate descent refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - 20 - elapsed
    if remaining > 10:
        max_rounds = min(30, max(5, int(remaining / 3)))
        optimized, opt_score = _greedy_flip_optimize(
            best_pattern,
            max_rounds=max_rounds,
            time_limit=remaining,
            start_time=start_time
        )
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Phase 5: Pairwise swap optimization
    elapsed = time.time() - start_time
    remaining = DEADLINE - 15 - elapsed
    if remaining > 10:
        optimized, opt_score = _pairwise_swap_optimize(
            best_pattern,
            max_rounds=8,
            time_limit=remaining,
            start_time=start_time
        )
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Phase 6: Final quick greedy pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        optimized, opt_score = _greedy_flip_optimize(
            best_pattern,
            max_rounds=10,
            time_limit=DEADLINE - 10 - elapsed,
            start_time=start_time
        )
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Ensure we return integer signs
    result = np.sign(best_pattern)
    result[result == 0] = 1
    return result.astype(int)
