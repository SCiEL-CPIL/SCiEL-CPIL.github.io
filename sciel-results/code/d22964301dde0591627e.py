"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math


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


def _autoconv_smooth_max(h, beta=200.0):
    """Compute smooth max of absolute autoconvolution using log-sum-exp."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    abs_ac = np.abs(ac)
    # Log-sum-exp for smooth max
    max_val = np.max(abs_ac)
    if max_val < 1e-10:
        return 0.0
    shifted = abs_ac - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted)) / len(abs_ac)) / beta
    return log_sum_exp


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


def _continuous_objective(x, beta=200.0):
    """Objective for continuous optimization: smooth max|autoconv| / (sum h)^2."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_smooth_max(h, beta) / (s * s)


def _m_sequence(n, poly=0x13, seed=0):
    """
    Generate an m-sequence (maximum length sequence) over GF(2) using a primitive polynomial.
    poly is the polynomial in binary (e.g., 0x13 = x^4+x+1 for degree 4).
    Returns a ±1 sequence of length n by tiling the m-sequence.
    """
    # Extract degree from polynomial
    deg = poly.bit_length() - 1
    if deg < 2:
        deg = 4
        poly = 0x13  # x^4 + x + 1
    
    # Initialize LFSR state (non-zero)
    state = 1  # Start with state 1
    
    # Generate 2^deg - 1 bits (one full period of m-sequence)
    period = (1 << deg) - 1
    bits = []
    
    for _ in range(period):
        bits.append(state & 1)
        # Compute next state: shift right, feedback from poly
        # The polynomial has bits set at positions corresponding to powers of x
        # For poly = sum of x^i, feedback = XOR of bits at those positions
        # Standard LFSR: new_bit = XOR of bits where poly has 1s (excluding the leading bit)
        feedback = 0
        for i in range(deg):
            if (poly >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << (deg - 1))
    
    # Convert bits to ±1 sequence
    mseq = np.array([1.0 if b == 1 else -1.0 for b in bits])
    
    # Tile to reach length n
    if len(mseq) >= n:
        return mseq[:n]
    else:
        reps = (n + len(mseq) - 1) // len(mseq)
        return np.tile(mseq, reps)[:n]


def _continuous_optimize(h_init, deadline=None):
    """
    Optimize in continuous space using scipy.optimize.minimize (L-BFGS-B),
    then quantize to ±1 and refine with local sign flips.
    This allows exploring the landscape more smoothly than pure sign flips.
    """
    h0 = h_init.astype(float).copy()
    
    # Ensure sum is not too small for stability
    s = np.sum(h0)
    if abs(s) < N * 0.1:
        if s < 0:
            h0 = -h0
        # Add a small bias to push away from zero sum, then normalize
        # We'll let the optimizer handle it, but avoid singularities
        if abs(np.sum(h0)) < 1e-6:
            h0[0] = 1.0

    best_h = h0.copy()
    best_val = _score(h0)  # Use exact score for tracking

    try:
        # Use L-BFGS-B with smooth surrogate for gradient-friendly optimization
        res = minimize(
            _continuous_objective,
            h0,
            method='L-BFGS-B',
            options={'maxiter': 80, 'ftol': 1e-8, 'maxfun': 300}
        )
        if res.x is not None and not np.any(np.isnan(res.x)):
            h_cont = res.x
            # Evaluate with exact score
            val_exact = _score(h_cont)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_cont
    except Exception:
        pass

    # Quantize the best continuous solution to ±1
    h_quant = np.sign(best_h)
    h_quant[h_quant == 0] = 1.0
    
    # Also try a biased quantization to maintain non-zero sum if needed
    s_q = np.sum(h_quant)
    if abs(s_q) < N * 0.1:
        if s_q < 0:
            h_quant = -h_quant
        if abs(np.sum(h_quant)) < 1e-6:
            h_quant[0] = 1.0

    # Refine with local sign flips
    h_opt, val_opt = _local_sign_flip(h_quant, max_passes=3, deadline=deadline)
    
    return h_opt, val_opt


def _chirp_sequence(n, k=1):
    """Generate a chirp-like sequence with parameter k."""
    x = np.arange(n)
    h = np.sign(np.cos(2 * np.pi * k * x * (x + 1) / (2 * n)))
    h[h == 0] = 1.0
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


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)
    
    # Phase 0: Try m-sequences first (algebraic structure)
    # Use different primitive polynomials for diversity
    mseq_polys = [0x13, 0x1B, 0x5D, 0x23, 0x25, 0x35, 0x45, 0x59]
    for poly in mseq_polys:
        if time.time() > deadline:
            break
        h_m = _m_sequence(N, poly=poly)
        # Try both the sequence and its negation
        for sign in [1.0, -1.0]:
            h_try = sign * h_m
            val_try = _score(h_try)
            if val_try < best_val:
                best_val = val_try
                best_h = h_try.copy()
        # Refine the m-sequence with local sign flips
        h_ref, val_ref = _local_sign_flip(h_m, max_passes=5, deadline=deadline)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 1: Diverse initializations with continuous optimization
    n_restarts_cont = 20
    
    for r in range(n_restarts_cont):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 7 == 0:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 7 == 1:
            # Low frequency cosine
            k = (r // 7) % 5 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 7 == 2:
            # Chirp
            h = _chirp_sequence(N, k=1 + (r // 7) * 0.5)
        elif r % 7 == 3:
            # Random continuous values in [-1, 1]
            h = np.random.uniform(-1, 1, size=N)
        elif r % 7 == 4:
            # Block pattern
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(5, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 7 == 5:
            # M-sequence based initialization
            h = _m_sequence(N, poly=mseq_polys[r % len(mseq_polys)])
        else:
            # M-sequence with random sign flips
            h = _m_sequence(N, poly=mseq_polys[(r + 3) % len(mseq_polys)])
            n_flips = np.random.randint(3, 15)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]

        # Continuous optimization
        h_opt, val_opt = _continuous_optimize(h, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 2: Refine the best solution found so far with more local sign flips
    # and try a few more discrete-only restarts if time permits
    if time.time() < deadline - 10:
        # Refine best_h further
        h_refined, val_refined = _local_sign_flip(best_h, max_passes=15, deadline=deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined

        # Try m-sequence-based discrete restarts
        for r in range(30):
            if time.time() > deadline:
                break
            poly = mseq_polys[r % len(mseq_polys)]
            h = _m_sequence(N, poly=poly)
            # Perturb with random flips
            n_flips = np.random.randint(2, 12)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
            
            h_opt, val_opt = _local_sign_flip(h, max_passes=5, deadline=deadline)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt

        # Try random discrete restarts
        n_restarts_disc = 30
        for r in range(n_restarts_disc):
            if time.time() > deadline:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            # Small perturbation: flip a few random bits
            n_flips = np.random.randint(1, 10)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
            
            h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt

    return best_h

# EVOLVE-BLOCK-END
