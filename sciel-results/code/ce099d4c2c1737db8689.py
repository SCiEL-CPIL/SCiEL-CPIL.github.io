"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def _autoconv_max_abs(x):
    """Compute max absolute autoconvolution value."""
    if np.all(x == 0):
        return 0.0
    ac = fftconvolve(x, x, mode='full')
    return np.max(np.abs(ac))


def _score(x):
    """max|autoconv| / (sum)^2 - lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p."""
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1


def _m_sequence(p):
    """Generate m-sequence (maximal length LFSR sequence) of length p-1 for prime p.
    Uses the simplest non-trivial primitive polynomial for small primes."""
    # For simplicity, use a known good construction
    # Actually, let's use a different approach: generate a sequence with good autocorrelation
    # using a simple LFSR
    
    # For prime p, we can use the sequence based on quadratic residues
    # This gives a Legendre sequence of length p-1
    seq = []
    for i in range(1, p):
        seq.append(_legendre_symbol(i, p))
    return np.array(seq, dtype=float)


def _local_sign_flip(x, rng, max_iters=200, time_limit=None, start_time=None):
    """Local search by flipping individual signs to reduce the score."""
    best_val = _score(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        # Try a random subset of positions
        trial_indices = rng.choice(N, size=min(30, N), replace=False)
        
        for idx in trial_indices:
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    break
            
            orig = best_x[idx]
            if orig == 0:
                continue
            best_x[idx] = -orig
            new_val = _score(best_x)
            
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                best_x[idx] = orig
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_x = np.ones(N)
    best_val = _score(best_x)
    
    # Strategy 1: Legendre-based sequences (quadratic residue sequences)
    # These have provably low off-peak autocorrelation
    
    # Try various primes near N
    primes_to_try = [397, 389, 383, 379, 373, 367, 359, 353, 349, 347]
    
    for p in primes_to_try:
        if time.time() - start_time > deadline - 30:
            break
        
        # Generate Legendre sequence of length p-1
        leg_seq = _m_sequence(p)
        
        # Pad or truncate to length N
        if len(leg_seq) >= N:
            x = leg_seq[:N].copy()
        else:
            x = np.zeros(N)
            x[:len(leg_seq)] = leg_seq
            # Fill remaining with random signs
            x[len(leg_seq):] = rng.choice([-1.0, 1.0], size=N - len(leg_seq))
        
        # Ensure non-zero sum
        if abs(np.sum(x)) < 1e-6:
            flip_idx = rng.choice(N, size=max(1, N // 10), replace=False)
            x[flip_idx] *= -1
        
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Local refinement
        x_refined, val_refined = _local_sign_flip(x, rng, max_iters=100, 
                                                    time_limit=deadline, start_time=start_time)
        if val_refined < best_val:
            best_val = val_refined
            best_x = x_refined.copy()
    
    # Strategy 2: Random sparse sequences with local search
    for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if time.time() - start_time > deadline - 20:
            break
        
        for r in range(3):
            if time.time() - start_time > deadline - 10:
                break
            
            mask = rng.random(N) < density
            signs = rng.choice([-1.0, 1.0], size=N)
            x = np.zeros(N)
            x[mask] = signs[mask]
            
            if abs(np.sum(x)) < 1e-6:
                flip_idx = rng.choice(N, size=max(1, N // 10), replace=False)
                x[flip_idx] *= -1
            
            val = _score(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
            
            x_refined, val_refined = _local_sign_flip(x, rng, max_iters=150,
                                                        time_limit=deadline, start_time=start_time)
            if val_refined < best_val:
                best_val = val_refined
                best_x = x_refined.copy()
    
    # Strategy 3: Continuous optimization via gradient descent on real-valued heights
    if time.time() - start_time < deadline - 15:
        # Start from best discrete solution and allow continuous values
        x_cont = best_x.astype(float).copy()
        
        for trial in range(50):
            if time.time() - start_time > deadline - 5:
                break
            
            # Try random perturbations
            idx = rng.choice(N, size=10, replace=False)
            for i in idx:
                orig = x_cont[i]
                # Try scaling
                factor = rng.uniform(0.3, 1.7)
                x_cont[i] = orig * factor
                new_val = _score(x_cont)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x_cont.copy()
                else:
                    x_cont[i] = orig
    
    # Ensure the result has reasonable sum
    if abs(np.sum(best_x)) < 1e-6:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
