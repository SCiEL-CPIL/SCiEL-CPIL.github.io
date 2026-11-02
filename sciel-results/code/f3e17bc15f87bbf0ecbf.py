import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _signed_local_search(x_init, max_iters=2000, rng=None):
    """
    Local search on signed ±1 sequences.
    Flip individual bits or small groups to minimize max|autoconv|/S^2.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Start with ±1 sequence
    x = np.sign(x_init)
    # Handle zeros: assign random sign
    zero_mask = x == 0
    if np.any(zero_mask):
        x[zero_mask] = rng.choice([-1, 1], size=np.sum(zero_mask))
    
    best_x = x.copy()
    best_val = _objective(best_x)
    
    no_improve_count = 0
    max_no_improve = 200
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if no_improve_count > max_no_improve:
            break
        
        # Try flipping a random subset of 1-3 bits
        n_flips = rng.integers(1, 4)
        if n_flips > N:
            n_flips = N
        flip_indices = rng.choice(N, size=n_flips, replace=False)
        
        trial = best_x.copy()
        trial[flip_indices] *= -1
        
        val = _objective(trial)
        
        if val < best_val:
            best_val = val
            best_x = trial.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_x, best_val

def _signed_local_search_targeted(x_init, max_iters=3000, rng=None):
    """
    Targeted local search: focus on reducing the current peak.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    x = np.sign(x_init)
    zero_mask = x == 0
    if np.any(zero_mask):
        x[zero_mask] = rng.choice([-1, 1], size=np.sum(zero_mask))
    
    best_x = x.copy()
    best_val = _objective(best_x)
    
    no_improve_count = 0
    max_no_improve = 300
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if no_improve_count > max_no_improve:
            break
        
        # Find the current peak location
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        
        # The peak is at ac[peak_idx] = sum_j x[j] * x[peak_idx - j] (with bounds)
        # Flipping x[j] changes ac[peak_idx] by -2*x[j]*x[peak_idx-j] if both in range
        # We want to flip j such that this reduces |ac[peak_idx]|
        
        # Compute for each j the change in ac[peak_idx] if we flip x[j]
        k = peak_idx
        changes = np.zeros(N)
        for j in range(N):
            val = 0.0
            if k - j >= 0 and k - j < N:
                val += best_x[k-j]
            if j - k >= 0 and j - k < N:
                val += best_x[j-k]
            # Flipping x[j] changes ac[k] by -2 * x[j] * (contribution from other term)
            # Actually: ac[k] = sum_i x[i]*x[k-i], flipping x[j] changes it by
            # -2*x[j]*x[k-j] if k-j in range, and -2*x[j]*x[j-k] if j-k in range
            # But we already computed the sum of those terms, so change = -2*x[j]*that_sum
            # Wait, that's not quite right. Let me think again.
            # ac[k] = sum_i x[i]*x[k-i]
            # If we flip x[j], the terms involving x[j] are:
            #   x[j]*x[k-j] (if k-j in [0,N))
            #   x[j-k]*x[j] (if j-k in [0,N)) which is x[j]*x[j-k]
            # So total contribution = x[j]*(x[k-j] + x[j-k]) where terms are 0 if out of range
            # Flipping x[j] to -x[j] changes this to -x[j]*(x[k-j] + x[j-k])
            # Change = -2*x[j]*(x[k-j] + x[j-k])
            # But we computed val = x[k-j] + x[j-k] (with bounds handling)
            # So change = -2*x[j]*val
            changes[j] = -2.0 * best_x[j] * val
        
        # We want to flip j that most reduces |ac[k]|
        # Current ac[k] has sign = sign(ac[k])
        # We want to change ac[k] in the direction that reduces its magnitude
        # If ac[k] > 0, we want negative change; if ac[k] < 0, we want positive change
        target_sign = -np.sign(ac[peak_idx])
        # Score each flip: how much it moves ac[k] in the desired direction
        scores = changes * target_sign
        # Pick the best few candidates
        top_k = min(5, N)
        top_indices = np.argsort(scores)[-top_k:]
        
        improved = False
        for idx in top_indices:
            trial = best_x.copy()
            trial[idx] *= -1
            val = _objective(trial)
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                no_improve_count = 0
                improved = True
                break
        
        if not improved:
            # Try random flips as fallback
            n_flips = rng.integers(1, 4)
            flip_indices = rng.choice(N, size=n_flips, replace=False)
            trial = best_x.copy()
            trial[flip_indices] *= -1
            val = _objective(trial)
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    return best_x, best_val

def _generate_random_signed(rng, N):
    """Generate a random ±1 sequence."""
    return rng.choice([-1, 1], size=N).astype(float)

def _generate_structured_signed(rng, N):
    """
    Generate structured ±1 sequences that may have good autoconvolution properties.
    Uses patterns inspired by Barker codes, Legendre sequences, etc.
    """
    seq_type = rng.integers(0, 5)
    
    if seq_type == 0:
        # Random
        return rng.choice([-1, 1], size=N).astype(float)
    elif seq_type == 1:
        # Alternating with random perturbations
        x = np.ones(N) * (-1)**np.arange(N)
        flip_mask = rng.random(N) < 0.1
        x[flip_mask] *= -1
        return x
    elif seq_type == 2:
        # Block pattern: random block lengths
        x = np.zeros(N)
        pos = 0
        while pos < N:
            block_len = rng.integers(1, 10)
            val = rng.choice([-1, 1])
            x[pos:pos+block_len] = val
            pos += block_len
        return x
    elif seq_type == 3:
        # Based on quadratic residue (Legendre-like)
        # Use a prime near N
        p = N
        while p > 2:
            # Check if p is prime
            if all(p % i != 0 for i in range(2, int(p**0.5)+1)):
                break
            p -= 1
        if p < 2:
            p = N
        x = np.zeros(N)
        for i in range(N):
            # Quadratic residue test mod p
            if i % p == 0:
                x[i] = 0  # Will be handled by sign
            else:
                # Euler's criterion: a^((p-1)/2) mod p = 1 if QR, -1 if NQR
                try:
                    if pow(i % p, (p-1)//2, p) == 1:
                        x[i] = 1
                    else:
                        x[i] = -1
                except:
                    x[i] = 1
        # Replace zeros
        zero_mask = x == 0
        if np.any(zero_mask):
            x[zero_mask] = rng.choice([-1, 1], size=np.sum(zero_mask))
        return x
    else:
        # Random with autocorrelation bias: start random, then do a few local search steps
        x = rng.choice([-1, 1], size=N).astype(float)
        # Do a few quick flips
        for _ in range(10):
            idx = rng.integers(0, N)
            trial = x.copy()
            trial[idx] *= -1
            if _objective(trial) < _objective(x):
                x = trial
        return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Strategy: Focus on signed ±1 sequences with local search
    # This is a structural change from continuous optimization
    
    n_restarts = 15
    for restart in range(n_restarts):
        if _check_deadline():
            break
        
        # Generate initial signed sequence
        if restart == 0:
            x0 = _generate_random_signed(rng, N)
        elif restart == 1:
            x0 = _generate_structured_signed(rng, N)
        elif restart == 2:
            x0 = _generate_structured_signed(rng, N)
        elif restart == 3:
            x0 = _generate_structured_signed(rng, N)
        else:
            x0 = _generate_random_signed(rng, N)
        
        # Local search phase 1: targeted
        x1, val1 = _signed_local_search_targeted(x0, max_iters=1000, rng=rng)
        if val1 < best_val:
            best_val = val1
            best_x = x1.copy()
        
        if _check_deadline():
            break
        
        # Local search phase 2: general
        x2, val2 = _signed_local_search(x1, max_iters=500, rng=rng)
        if val2 < best_val:
            best_val = val2
            best_x = x2.copy()
        
        if _check_deadline():
            break
        
        # Try a few more targeted rounds
        x3, val3 = _signed_local_search_targeted(x2, max_iters=500, rng=rng)
        if val3 < best_val:
            best_val = val3
            best_x = x3.copy()
    
    # Also try: take the best continuous solution and round to signs
    # But we don't have one from this version, so skip
    
    return best_x
