"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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


def _legendre_sequence(n):
    """Generate a Legendre symbol sequence of length n.
    
    Uses the Legendre symbol (x/p) for a prime p close to n.
    For x = 0, 1, ..., n-1, the sequence is (x/p) where p is the smallest prime >= n.
    This gives a sequence with good autocorrelation properties (Burgess bound).
    """
    # Find smallest prime >= n
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Compute Legendre symbols for x = 0, 1, ..., n-1 mod p
    # (x/p) = x^((p-1)/2) mod p, which is 0 if x=0, 1 if quadratic residue, -1 otherwise
    half = (p - 1) // 2
    h = np.zeros(n)
    for x in range(n):
        if x == 0:
            h[x] = 1.0  # Convention: use +1 for x=0
        else:
            # Compute x^((p-1)/2) mod p
            val = pow(x, half, p)
            if val == 1:
                h[x] = 1.0
            elif val == p - 1:  # -1 mod p
                h[x] = -1.0
            else:
                h[x] = 1.0  # Should not happen for prime p and 0 < x < p
    
    return h


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


def _continuous_objective(x):
    """Objective for continuous optimization: max|autoconv| / (sum h)^2."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


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
    best_val = _continuous_objective(h0)

    try:
        # Use L-BFGS-B which handles bounds well
        res = minimize(
            _continuous_objective,
            h0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-6, 'maxfun': 200}
        )
        if res.x is not None and not np.any(np.isnan(res.x)):
            h_cont = res.x
            val_cont = _continuous_objective(h_cont)
            if val_cont < best_val:
                best_val = val_cont
                best_h = h_cont
    except Exception:
        pass

    # Quantize the best continuous solution to ±1
    # We can try different quantization thresholds or just sign
    h_quant = np.sign(best_h)
    # Handle zeros
    h_quant[h_quant == 0] = 1.0
    
    # Also try a biased quantization to maintain non-zero sum if needed
    s_q = np.sum(h_quant)
    if abs(s_q) < N * 0.1:
        # Flip the whole thing if sum is negative
        if s_q < 0:
            h_quant = -h_quant
        # If still small, force some elements to +1
        # Simple heuristic: set the elements with largest positive value to +1
        # But since it's ±1, we just pick a random subset to flip to +1
        # Or better, just use the continuous values to guide a soft sign
        # For now, just ensure it's not zero
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
    
    # Phase 1: Diverse initializations with continuous optimization
    # We use a smaller number of restarts for continuous opt as it's more expensive
    n_restarts_cont = 20
    
    # Start with Legendre sequence as it has provably good autocorrelation
    h_legendre = _legendre_sequence(N)
    h_opt, val_opt = _local_sign_flip(h_legendre, max_passes=10, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    for r in range(n_restarts_cont):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 5 == 0:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Low frequency cosine
            k = (r // 5) % 5 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 5 == 2:
            # Chirp
            h = _chirp_sequence(N, k=1 + (r // 5) * 0.5)
        elif r % 5 == 3:
            # Random continuous values in [-1, 1]
            h = np.random.uniform(-1, 1, size=N)
        else:
            # Block pattern
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(5, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len

        # Continuous optimization
        h_opt, val_opt = _continuous_optimize(h, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 2: Refine the best solution found so far with more local sign flips
    # and try a few more discrete-only restarts if time permits
    if time.time() < deadline - 10:
        # Refine best_h further
        h_refined, val_refined = _local_sign_flip(best_h, max_passes=10, deadline=deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined

        # Try a few more discrete restarts
        n_restarts_disc = 50
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

    # Phase 3: Final high-precision polish pass with more iterations
    if time.time() < deadline - 5:
        h_final, val_final = _local_sign_flip(best_h, max_passes=20, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    return best_h

# EVOLVE-BLOCK-END
