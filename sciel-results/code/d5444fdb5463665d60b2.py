"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))


def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val


def _soft_max_abs(ac, beta=10.0):
    """Compute soft-max of |ac| using log-sum-exp for smoothness."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m == 0:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (abs_ac - m))))
    return m + np.log(np.sum(np.exp(beta * (abs_ac - m)))) / beta


def _score_soft(h, beta=10.0):
    """Compute soft objective: soft_max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (s * s)


def _smooth_optimize(h, deadline=None, max_iter=500, lr=0.01):
    """Smooth gradient-free optimization using random perturbations on soft objective."""
    h = h.astype(float).copy()
    best_val = _score_soft(h)
    best_h = h.copy()
    
    rng = np.random.RandomState(42)
    step_size = 0.1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random perturbation
        pert = rng.normal(0, step_size, size=N)
        h_new = h + pert
        
        # Keep values in reasonable range
        h_new = np.clip(h_new, -2, 2)
        
        val_new = _score_soft(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new.copy()
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Decrease step size on rejection
            step_size = max(step_size * 0.95, 0.01)
    
    return best_h, best_val


def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p."""
    if n % p == 0:
        return 0
    return pow(n, (p - 1) // 2, p) - 1 if pow(n, (p - 1) // 2, p) == 1 else -1


def _legendre_sequence(n, p=None):
    """Generate a Legendre sequence of length n using prime p."""
    if p is None:
        # Find a prime >= n
        p = n
        while True:
            is_prime = True
            for i in range(2, int(np.sqrt(p)) + 1):
                if p % i == 0:
                    is_prime = False
                    break
            if is_prime:
                break
            p += 1
    h = np.zeros(n)
    for i in range(n):
        sym = _legendre_symbol(i + 1, p)
        if sym == 0:
            h[i] = 1.0  # Convention for 0
        else:
            h[i] = float(sym)
    return h


def _chirp_sequence(n):
    """Generate a chirp (Zadoff-Chu-like) sequence."""
    x = np.arange(n)
    h = np.sign(np.cos(2 * np.pi * x * (x + 1) / (2 * n)))
    return h


def _barker_like(n):
    """Generate a Barker-like sequence of length n using known Barker codes or random ±1."""
    barkers = {
        2: [1, 1],
        3: [1, 1, -1],
        4: [1, 1, -1, 1],
        5: [1, 1, -1, -1, 1],
        7: [1, -1, -1, 1, -1, 1, 1],
        11: [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    }
    if n in barkers:
        return np.array(barkers[n], dtype=float)
    return np.random.choice([-1.0, 1.0], size=n)


def _m_sequence_like(n, seed=42):
    """Generate an m-sequence-like pattern using a simple LFSR or just structured random."""
    rng = np.random.RandomState(seed)
    h = np.ones(n)
    for i in range(n):
        if (i & 0x5555) != 0:
            h[i] = -1.0
    return h


def _polish_discrete(h, deadline=None, max_passes=20):
    """Final high-precision polish: more passes of sign-flip with pair flips."""
    h = h.copy()
    best_val = _score(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Single flips
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        
        # Pair flips
        if not improved:
            break
        for i in range(N - 1):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            h[i+1] = -h[i+1]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
                h[i+1] = -h[i+1]
        
        if not improved:
            break
    
    return h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)
    n_restarts = 200

    for r in range(n_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 10 == 0:
            # Legendre sequence - strong algebraic structure
            h = _legendre_sequence(N)
        elif r % 10 == 1:
            # Legendre with random shift
            h = _legendre_sequence(N)
            shift = np.random.randint(0, N)
            h = np.roll(h, shift)
        elif r % 10 == 2:
            # Legendre with random sign flips on a few elements
            h = _legendre_sequence(N)
            n_flips = np.random.randint(1, 10)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
        elif r % 10 == 3:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 10 == 4:
            # Mostly +1 with some random -1
            h = np.ones(N)
            n_neg = np.random.randint(N // 4, N // 2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 10 == 5:
            # Chirp-like pattern
            h = _chirp_sequence(N)
        elif r % 10 == 6:
            # Random with continuous values
            h = np.random.uniform(-1, 1, size=N)
        elif r % 10 == 7:
            # Block pattern with random block sizes
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(1, N // 10 + 1)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 10 == 8:
            # M-sequence-like pattern
            h = _m_sequence_like(N, seed=r * 7 + 1)
        else:
            # Barker-like tiling
            h = np.ones(N)
            pos = 0
            while pos < N:
                blen = np.random.choice([3, 4, 5, 7, 11, 13])
                if pos + blen > N:
                    blen = N - pos
                if blen in [3, 4, 5, 7, 11, 13]:
                    code = _barker_like(blen)
                    if np.random.rand() < 0.5:
                        code = -code
                    h[pos:pos + blen] = code
                else:
                    h[pos:pos + blen] = np.random.choice([-1.0, 1.0], size=blen)
                pos += blen

        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = h + 0.3
            h = np.sign(h)

        # Phase 1: Discrete local sign-flip optimization
        h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)

        # Phase 2: Smooth continuous optimization on the soft surrogate
        if time.time() < deadline - 10:
            h_smooth, val_smooth = _smooth_optimize(h_opt, deadline=deadline, max_iter=200)
            # Convert back to discrete ±1
            h_disc = np.sign(h_smooth)
            # If all zeros, use original
            if np.all(h_disc == 0):
                h_disc = h_opt
            else:
                h_opt, val_opt = _local_sign_flip(h_disc, max_passes=3, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Final polish with more iterations
    if time.time() < deadline - 5:
        best_h, best_val = _polish_discrete(best_h, deadline=deadline, max_passes=30)

    return best_h
# EVOLVE-BLOCK-END
