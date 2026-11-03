"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly N/2,
    minimizing the maximum overlap of h with shifted 1-h.
    
    Overlap at lag k is defined as: sum_i h_i * (1 - h[(i+k) % N])
    We want to minimize max_k overlap(k).
    
    Key insight: overlap(k) = N/2 - sum_i h_i * h[(i+k)%N]
    So minimizing max overlap = maximizing min autocorrelation.
    
    The uniform solution h_i = 0.5 gives overlap(k) = N/4 for all k.
    
    We can do better by using structured patterns. Let's think about what
    minimizes the maximum overlap.
    
    Actually, let's use a different approach: we want h to be as "anti-correlated" 
    with its shifts as possible, but constrained to [0,1] with mean 0.5.
    
    Let's try a local search approach with a deadline guard.
    """
    deadline = time.time() + 110  # 10 second safety margin
    
    # Start with uniform
    h = np.full(N, 0.5)
    
    def compute_overlap(h):
        """Compute max over all lags k of sum_i h_i * (1 - h[(i+k)%N])"""
        # Use FFT for fast computation of all lags
        # overlap(k) = sum_i h_i * (1 - h_{i+k}) = N/2 - sum_i h_i * h_{i+k}
        # The autocorrelation can be computed via FFT
        h_ext = np.concatenate([h, h])
        # For each lag k, sum_i h[i] * h[i+k] for i=0..N-1
        # This is the circular autocorrelation
        H = np.fft.fft(h)
        auto = np.fft.ifft(H * np.conj(H)).real
        # auto[k] = sum_i h[i] * h[(i-k)%N] for k=0..N-1
        # overlap(k) = N/2 - auto[k]
        overlaps = N/2.0 - auto
        return np.max(overlaps)
    
    best_h = h.copy()
    best_score = compute_overlap(best_h)
    
    # Local search: try random perturbations
    rng = np.random.RandomState(42)
    step_size = 0.1
    
    while time.time() < deadline:
        # Generate a random perturbation
        dh = rng.randn(N) * step_size
        # Project to keep sum = N/2 and in [0,1]
        h_trial = h + dh
        # Adjust to maintain sum constraint
        diff = h_trial.sum() - N/2.0
        h_trial -= diff / N
        # Clip to [0, 1]
        h_trial = np.clip(h_trial, 0, 1)
        # Re-adjust sum if clipping changed it
        diff = h_trial.sum() - N/2.0
        h_trial -= diff / N
        h_trial = np.clip(h_trial, 0, 1)
        
        score = compute_overlap(h_trial)
        if score < best_score:
            best_score = score
            best_h = h_trial.copy()
        
        # Update h (accept if better or with some probability)
        if score <= best_score or rng.random() < 0.3:
            h = h_trial.copy()
        
        # Adapt step size
        if score < best_score:
            step_size = max(0.01, step_size * 0.95)
        else:
            step_size = min(0.5, step_size * 1.05)
    
    # Ensure exact sum and bounds
    diff = best_h.sum() - N/2.0
    best_h -= diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
