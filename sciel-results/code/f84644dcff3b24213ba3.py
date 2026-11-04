"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _fast_autoconv(f):
    """Compute full autoconvolution using FFT for speed."""
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    AC = np.fft.irfft(F * F, n=n_fft)
    return AC[:2 * N - 1]

def _score(f):
    """Calculate the score: max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-10:
        return 1e10
    ac = _fast_autoconv(f)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def construct():
    """
    Heuristic search over deterministic candidates.
    Theoretical analysis suggests that constant vectors (like all ones) minimize the ratio
    because they maximize the sum squared relative to the peak autoconvolution.
    We evaluate a few simple candidates to ensure robustness.
    """
    candidates = []
    
    # Candidate 1: All ones (Maximizes sum, minimizes relative peak)
    candidates.append(np.ones(N))
    
    # Candidate 2: All -1 (Same score as all ones)
    candidates.append(-np.ones(N))
    
    # Candidate 3: Alternating 1, -1 (Sum is 0 or 1, bad score usually)
    # Just in case "signed" implies strict alternation or something, though unlikely to be optimal.
    candidates.append(np.array([1.0, -1.0] * (N // 2) + [1.0] * (N % 2)))
    
    # Candidate 4: Random with positive bias (Sum > 0, but AC peak might be higher relative to sum?)
    rng = np.random.default_rng(42)
    # Bias towards 1 to keep sum large
    x_rand = rng.choice([-1.0, 1.0], size=N, p=[0.1, 0.9])
    candidates.append(x_rand)
    
    # Candidate 5: A "block" sequence? e.g., half 1s, half -1s?
    # Sum = 0, so score is infinite. Avoid.
    
    # Evaluate all candidates
    best_score = 1e10
    best_f = candidates[0]
    
    for f in candidates:
        s_val = _score(f)
        if s_val < best_score:
            best_score = s_val
            best_f = f
            
    return best_f
# EVOLVE-BLOCK-END
