"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _generate_m_sequence(n, poly, init):
    """Generate an m-sequence of length n using a linear feedback shift register."""
    # poly is a list of tap positions (0-indexed from LSB)
    # init is the initial state (integer)
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
    return np.array(seq[:n])


def _try_m_sequences(deadline, rng):
    """Try various m-sequence based patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Various LFSR polynomials for different lengths
    polys = [
        [1, 0],  # period 3
        [2, 0],  # period 7
        [3, 1],  # period 15
        [4, 1],  # period 31
        [5, 2],  # period 63
        [6, 1],  # period 127
        [7, 3],  # period 255
        [8, 4],  # period 511
        [9, 5],  # period 1023
        [10, 7], # period 2047
    ]
    
    for poly in polys:
        if time.time() > deadline:
            break
        # Try different initial states
        for init in range(min(10, (1 << len(poly)) - 1)):
            if time.time() > deadline:
                break
            seq = _generate_m_sequence(N, poly, init + 1)
            v = _metric(seq)
            if v < best_val:
                best_val = v
                best_x = seq.copy()
            
            # Also try negated version
            seq_neg = -seq
            v = _metric(seq_neg)
            if v < best_val:
                best_val = v
                best_x = seq_neg.copy()
    
    return best_x, best_val


def _try_block_patterns(deadline, rng):
    """Try patterns with blocks of +1 and -1 of various sizes."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Try block sizes
    for block_size in [2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200]:
        if time.time() > deadline:
            break
        if block_size > N:
            continue
        
        n_blocks = N // block_size
        remainder = N % block_size
        
        # Pattern: alternating blocks
        x = np.zeros(N)
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            if i % 2 == 0:
                x[start:end] = 1.0
            else:
                x[start:end] = -1.0
        if remainder > 0:
            x[n_blocks * block_size:] = 1.0 if (n_blocks % 2 == 0) else -1.0
        
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
        
        # Pattern: random block assignment
        for _ in range(5):
            if time.time() > deadline:
                break
            x = np.zeros(N)
            for i in range(n_blocks):
                start = i * block_size
                end = start + block_size
                val = rng.choice([-1.0, 1.0])
                x[start:end] = val
            if remainder > 0:
                x[n_blocks * block_size:] = rng.choice([-1.0, 1.0])
            
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    
    return best_x, best_val


def _try_gaussian_smoothed(deadline, rng):
    """Try sequences that are smoothed versions of random patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    for sigma in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
        if time.time() > deadline:
            break
        for _ in range(10):
            if time.time() > deadline:
                break
            # Start with random ±1
            x_raw = rng.choice([-1.0, 1.0], size=N)
            # Smooth it
            from scipy.ndimage import gaussian_filter1d
            x_smooth = gaussian_filter1d(x_raw, sigma=sigma)
            # Normalize to have unit RMS
            rms = np.sqrt(np.mean(x_smooth**2))
            if rms > 1e-12:
                x_smooth = x_smooth / rms
            
            v = _metric(x_smooth)
            if v < best_val:
                best_val = v
                best_x = x_smooth.copy()
    
    return best_x, best_val


def _try_local_search(x, deadline, rng, max_rounds=100000):
    """Improved local search with adaptive step size."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 1.0
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Adaptive: reduce step size if stuck
        if no_improve > 1000:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.01:
                break
        
        # Try flipping k random positions with random magnitudes
        k = rng.integers(1, min(8, N//4) + 1)
        indices = rng.choice(N, size=k, replace=False)
        x_flipped = best_x.copy()
        
        for idx in indices:
            # Random perturbation
            delta = rng.normal(0, 0.5 * step_scale)
            x_flipped[idx] += delta
        
        v = _metric(x_flipped)
        if v < best_val:
            best_val = v
            best_x = x_flipped
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _try_simulated_annealing(x, deadline, rng, max_rounds=50000):
    """Simulated annealing for escaping local minima."""
    best_x = x.copy()
    best_val = _metric(best_x)
    current_x = x.copy()
    current_val = _metric(current_x)
    
    T_start = 0.1
    T_end = 0.001
    cooling = (T_end / T_start) ** (1.0 / max_rounds)
    T = T_start
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Random perturbation
        k = rng.integers(1, min(5, N//4) + 1)
        indices = rng.choice(N, size=k, replace=False)
        x_new = current_x.copy()
        x_new[indices] *= -1
        
        v_new = _metric(x_new)
        delta = v_new - current_val
        
        # Accept or reject
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_x = x_new
            current_val = v_new
            if v_new < best_val:
                best_val = v_new
                best_x = x_new.copy()
        
        T *= cooling
    
    return best_x, best_val


def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 130.0

    # Start from all-ones (Cauchy-Schwarz lower bound)
    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Phase 1: Try m-sequences
    if time.time() < deadline - 80:
        x_m, val_m = _try_m_sequences(deadline - 80, rng)
        if val_m < best_val:
            best_val = val_m
            best_x = x_m

    # Phase 2: Try block patterns
    if time.time() < deadline - 60:
        x_block, val_block = _try_block_patterns(deadline - 60, rng)
        if val_block < best_val:
            best_val = val_block
            best_x = x_block

    # Phase 3: Try Gaussian smoothed patterns
    if time.time() < deadline - 40:
        x_gauss, val_gauss = _try_gaussian_smoothed(deadline - 40, rng)
        if val_gauss < best_val:
            best_val = val_gauss
            best_x = x_gauss

    # Phase 4: Try random ±1 sequences
    if time.time() < deadline - 20:
        for _ in range(30):
            if time.time() > deadline - 20:
                break
            cand = rng.choice([-1.0, 1.0], size=N)
            v = _metric(cand)
            if v < best_val:
                best_val = v
                best_x = cand.copy()

    # Phase 5: Simulated annealing from best candidate
    if time.time() < deadline - 10:
        best_x, best_val = _try_simulated_annealing(best_x, deadline - 10, rng, max_rounds=30000)

    # Phase 6: Local search from best candidate
    if time.time() < deadline - 2:
        best_x, best_val = _try_local_search(best_x, deadline - 2, rng, max_rounds=50000)

    return best_x
# EVOLVE-BLOCK-END
