"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400
_DEADLINE = time.time() + 115.0

def _autoconv_max_abs(y, integral):
    """Compute max absolute autoconvolution value and return the ratio."""
    # Linear autocorrelation via FFT
    n = len(y)
    size = 2 * n
    Y = np.fft.rfft(y, size)
    ac = np.fft.irfft(Y * np.conj(Y), size)[:n]
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _construct_bias(bias_frac):
    """Construct a biased +/-1 sequence with the given fraction of +1s.
    Returns (sequence, score)."""
    if bias_frac <= 0:
        bias_frac = 0.0
    if bias_frac >= 1:
        bias_frac = 1.0
    
    n_plus = int(round(bias_frac * N))
    # Ensure n_plus is in [1, N] to avoid zero integral
    n_plus = max(1, min(N, n_plus))
    
    seq = np.ones(N)
    # Place -1s in a spread-out pattern to reduce autoconv sidelobes
    # Use evenly spaced positions for the -1s
    n_minus = N - n_plus
    if n_minus > 0:
        # Spread the -1s as evenly as possible
        positions = np.linspace(0, N - 1, n_minus, dtype=int)
        # Avoid exact duplicates if N is small
        if len(positions) != len(np.unique(positions)):
            # Fallback: take first n_minus positions
            positions = np.arange(n_minus)
        seq[positions] = -1
    
    integral = np.sum(seq)
    if abs(integral) < 1e-10:
        return seq, np.inf
    
    score = _autoconv_max_abs(seq, integral)
    return seq, score

def _local_refine(seq, max_iters=200):
    """Local refinement: flip individual bits to reduce the ratio."""
    best_seq = seq.copy()
    best_integral = np.sum(best_seq)
    best_score = _autoconv_max_abs(best_seq, best_integral)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(N):
            if time.time() > _DEADLINE:
                break
            # Try flipping position i
            old_val = best_seq[i]
            new_val = -old_val
            test_seq = best_seq.copy()
            test_seq[i] = new_val
            test_integral = best_integral + (new_val - old_val)
            if abs(test_integral) < 1e-10:
                continue
            test_score = _autoconv_max_abs(test_seq, test_integral)
            if test_score < best_score * (1 - 1e-10):
                best_seq[i] = new_val
                best_integral = test_integral
                best_score = test_score
                improved = True
        if not improved:
            break
    
    return best_seq, best_score

def _differential_evolution_simple(pop_size, max_gen, pop):
    """Simple differential evolution over bias fraction and a few pattern parameters."""
    # Parameters: [bias_frac, spacing_param]
    # We'll use a simple approach: evolve the bias and the pattern
    best_overall = np.inf
    best_seq = None
    
    for gen in range(max_gen):
        if time.time() > _DEADLINE:
            break
        for idx, params in enumerate(pop):
            if time.time() > _DEADLINE:
                break
            bias_frac = params[0]
            seq, score = _construct_bias(bias_frac)
            
            if score < best_overall:
                best_overall = score
                best_seq = seq.copy()
            
            # Update population (simple: replace worst with improved version)
            # For simplicity, just track the best
    
    return best_seq, best_overall

def construct():
    """Main entry point: construct a signed step sequence minimizing
    max|autoconv| / (integral)^2."""
    start_time = time.time()
    
    best_seq = None
    best_score = np.inf
    
    # Phase 1: Coarse search over bias fraction
    # The optimal bias is likely high (many +1s) to maximize the denominator
    # while keeping sidelobes low.
    
    bias_candidates = np.linspace(0.5, 1.0, 20)
    
    for bias in bias_candidates:
        if time.time() > _DEADLINE - 30:
            break
        seq, score = _construct_bias(bias)
        if score < best_score:
            best_score = score
            best_seq = seq.copy()
    
    # Phase 2: Local refinement on the best candidate
    if best_seq is not None:
        if time.time() < _DEADLINE - 20:
            best_seq, best_score = _local_refine(best_seq, max_iters=100)
    
    # Phase 3: Try a few more bias values near the best and refine
    if best_seq is not None and time.time() < _DEADLINE - 10:
        # Find the bias that produced the best result
        for bias in np.linspace(0.7, 0.95, 10):
            if time.time() > _DEADLINE - 5:
                break
            seq, score = _construct_bias(bias)
            if score < best_score:
                best_seq = seq.copy()
                best_score = score
            # Quick local refinement
            if time.time() < _DEADLINE - 10:
                seq, score = _local_refine(seq, max_iters=50)
                if score < best_score:
                    best_score = score
                    best_seq = seq.copy()
    
    # Phase 4: Final local refinement with more iterations
    if best_seq is not None and time.time() < _DEADLINE - 5:
        best_seq, best_score = _local_refine(best_seq, max_iters=200)
    
    if best_seq is None:
        # Fallback: all ones
        best_seq = np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END
