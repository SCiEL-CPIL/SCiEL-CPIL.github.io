"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_sidelobe_max(h):
    """Compute max |R(k)| for k=1..N-1 where R is autocorrelation."""
    # R(k) = sum_{i=0}^{N-1-k} h[i] * h[i+k]
    # Use FFT for efficiency
    n = len(h)
    # Zero-pad to 2*n for linear convolution
    f = np.fft.fft(h, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # acf[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k] for k=0..n-1
    # Sidelobes are k=1..n-1
    return np.max(np.abs(acf[1:]))

def _evaluate(h):
    """Score: upper bound on C3, lower is better."""
    return _autocorr_sidelobe_max(h)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with a random ±1 sequence
    rng = np.random.RandomState(42)
    h = rng.choice([-1, 1], size=N)
    
    best_h = h.copy()
    best_score = _evaluate(best_h)
    
    # Local search: flip individual bits to reduce max sidelobe
    # Use a deadline guard
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            h[i] = -h[i]
            score = _evaluate(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]  # revert
        
        if not improved:
            # Try random restarts or perturbations
            # Perturb: flip a few random bits
            num_flips = rng.randint(1, 5)
            flip_indices = rng.choice(N, size=num_flips, replace=False)
            for idx in flip_indices:
                h[idx] = -h[idx]
            score = _evaluate(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                improved = True
            else:
                # Revert
                for idx in flip_indices:
                    h[idx] = -h[idx]
            
            if not improved:
                # Full random restart
                h = rng.choice([-1, 1], size=N)
                score = _evaluate(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
