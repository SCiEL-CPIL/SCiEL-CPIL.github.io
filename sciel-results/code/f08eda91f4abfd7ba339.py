"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(h):
    """Compute max overlap of h with shifted 1-h."""
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    max_overlap = 0.0
    for s in range(1, N):
        shifted = np.roll(one_minus_h, s)
        overlap = np.dot(h, shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with constant
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _score(best_h)
    
    # Try low-frequency cosine with small amplitude
    # The key insight: we want to maximize min autocorrelation for s!=0
    # Using h[i] = 0.5 + a*cos(2*pi*k*i/N) gives R(s) = N/4 + (a^2*N/2)*cos(2*pi*k*s/N)
    # Min R(s) for s!=0 is N/4 - a^2*N/2, so max overlap = N/4 + a^2*N/2
    # This is always >= N/4, so single cosine doesn't help directly.
    # 
    # Instead, let's try a two-level sequence optimized for minimal max overlap.
    # Use a greedy/local search approach.
    
    # Try a structured approach: alternate blocks
    # h[i] = 0.5 + a * sign(sin(2*pi*i/N))  - but this has discontinuities
    
    # Let's try gradient-free local search with cosine basis
    # Represent h as 0.5 + sum of a few cosine terms
    # Optimize to minimize max overlap
    
    def h_from_params(params, freqs):
        """Build h from amplitude params and fixed frequencies."""
        h = np.full(N, 0.5)
        for a, f in zip(params, freqs):
            h += a * np.cos(2 * np.pi * f * np.arange(N) / N)
        return np.clip(h, 0, 1)
    
    # Try with a few low frequencies
    freqs = [1, 2, 3]
    
    # Simple coordinate descent on amplitudes
    params = np.zeros(len(freqs))
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        for j in range(len(freqs)):
            # Try to adjust params[j]
            old_val = params[j]
            # Binary search for optimal value in [-0.3, 0.3]
            best_p = old_val
            best_s = _score(h_from_params(params, freqs))
            for trial in np.linspace(-0.3, 0.3, 20):
                params[j] = trial
                s = _score(h_from_params(params, freqs))
                if s < best_s:
                    best_s = s
                    best_p = trial
            params[j] = best_p
            if best_p != old_val:
                improved = True
        if not improved:
            break
    
    h_candidate = h_from_params(params, freqs)
    # Ensure sum is exactly N/2
    diff = N/2 - np.sum(h_candidate)
    h_candidate += diff / N
    h_candidate = np.clip(h_candidate, 0, 1)
    # Re-normalize
    h_candidate *= (N/2) / np.sum(h_candidate)
    h_candidate = np.clip(h_candidate, 0, 1)
    
    s_candidate = _score(h_candidate)
    if s_candidate < best_score:
        best_h = h_candidate.copy()
        best_score = s_candidate
    
    # Try a different approach: use a binary-like sequence
    # h[i] = 1 for i in some set S, 0 otherwise, |S| = N/2
    # Optimize S to minimize max overlap
    
    # Use a simple heuristic: start with alternating, then local search
    h_bin = np.zeros(N)
    h_bin[::2] = 1.0  # 50 ones, 50 zeros
    
    s_bin = _score(h_bin)
    if s_bin < best_score:
        best_h = h_bin.copy()
        best_score = s_bin
    
    # Local search on binary sequence
    for iteration in range(2000):
        if time.time() > deadline:
            break
        # Try swapping a 1 with a 0
        ones = np.where(h_bin == 1)[0]
        zeros = np.where(h_bin == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
        s_new = _score(h_bin)
        if s_new >= best_score:
            h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
        else:
            best_h = h_bin.copy()
            best_score = s_new
    
    # Final: ensure sum is exactly N/2
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
