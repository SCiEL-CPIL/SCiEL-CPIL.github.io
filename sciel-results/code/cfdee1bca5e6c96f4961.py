"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.fft as fft


def _autoconvolve(x):
    """Compute autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = fft.rfft(x, size)
    ac = fft.irfft(X * X.conj(), size)
    return ac


def _objective(x):
    """Compute the ratio: max|autoconv| / (sum(x))^2. Lower is better."""
    total = np.sum(x)
    if abs(total) < 1e-12:
        return np.inf
    ac = _autoconvolve(x)
    return np.max(np.abs(ac)) / (total * total)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s margin, leaves 10s buffer
    
    # Start with ones (score 1.0)
    x = np.ones(N, dtype=np.float64)
    best_x = x.copy()
    best_score = _objective(best_x)
    
    # Use a gradient-free approach with structured perturbations
    # Try various sign-flip patterns to reduce max|autoconv|
    
    # Strategy: Try random sign assignments and local search
    # The autoconvolution of a ±1 sequence relates to difference sets
    
    # First: try a few structured patterns
    candidates = []
    
    # Pattern 1: Alternating signs
    x_alt = np.ones(N)
    x_alt[1::2] = -1
    candidates.append(x_alt)
    
    # Pattern 2: First half +1, second half -1
    x_half = np.ones(N)
    x_half[N//2:] = -1
    candidates.append(x_half)
    
    # Pattern 3: Blocks of varying size
    for block in [2, 4, 8, 16]:
        x_block = np.ones(N)
        x_block[1::block] = -1
        candidates.append(x_block)
    
    # Evaluate structured candidates
    for xc in candidates:
        sc = _objective(xc)
        if sc < best_score:
            best_score = sc
            best_x = xc.copy()
    
    # Now do iterative sign-flip local search with multiple restarts
    rng = np.random.default_rng(42)
    
    max_iterations = 5000
    flips_per_iter = 10
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Initialize: random signs or perturb best
        if restart == 0:
            x = best_x.copy()
        else:
            x = rng.choice([-1.0, 1.0], size=N)
            # Bias toward best solution
            if rng.random() < 0.5:
                x = best_x.copy()
                # Flip some random positions
                n_flip = rng.integers(10, 100)
                flip_idx = rng.choice(N, size=n_flip, replace=False)
                x[flip_idx] *= -1
        
        x_score = _objective(x)
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try flipping a small set of random positions
            n_flip = rng.integers(1, 5)
            flip_idx = rng.choice(N, size=n_flip, replace=False)
            
            x_trial = x.copy()
            x_trial[flip_idx] *= -1
            trial_score = _objective(x_trial)
            
            if trial_score < x_score:
                x = x_trial
                x_score = trial_score
                if x_score < best_score:
                    best_score = x_score
                    best_x = x.copy()
            else:
                # Occasionally accept worse solutions (simulated annealing lite)
                if rng.random() < 0.1:
                    x = x_trial
                    x_score = trial_score
        
        # Also try targeted: flip positions that contribute most to max|ac|
        if time.time() < deadline:
            ac = _autoconvolve(best_x)
            max_idx = np.argmax(np.abs(ac))
            # The autoconvolution at lag k is sum_i x[i]*x[i+k]
            # To reduce |ac[k]|, we want to flip signs that make terms cancel
            # Try flipping pairs around the peak
            for _ in range(100):
                if time.time() > deadline:
                    break
                x_trial = best_x.copy()
                # Flip a random subset
                n_flip = rng.integers(1, 15)
                flip_idx = rng.choice(N, size=n_flip, replace=False)
                x_trial[flip_idx] *= -1
                ts = _objective(x_trial)
                if ts < best_score:
                    best_score = ts
                    best_x = x_trial.copy()
    
    return best_x
# EVOLVE-BLOCK-END
