"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import correlate1d

N = 400
# Interval width: total length is 1/2, so each interval has width (1/2)/400
dx = 0.5 / N

# Deadline guard: keep a safety margin before the 120s/150s limit
_DEADLINE = 120.0 - 10.0  # 110s safety margin

def _autoconvolution_upper_bound(signed_steps):
    """
    Compute the upper bound on C3 from a signed step function.
    
    The signed step function f(x) takes values in {-1, 1} on 400 equal intervals
    on [-1/4, 1/4]. The autoconvolution is (f * f)(t) = ∫ f(x) f(x-t) dx.
    
    For a step function with constant values on intervals of width dx, the
    autoconvolution at lag k (in interval units) is:
        (f*f)(k*dx) = dx * sum_i f[i] * f[i+k]  (with appropriate bounds)
    
    The "upper bound on C3" is interpreted as the maximum absolute value of
    the autoconvolution over all lags (excluding the peak at lag 0, which is
    always N*dx = 1/2).
    
    Actually, let's reconsider. The task says "score is the upper bound on C3"
    and "lower is better". For a signed step function, the autoconvolution's
    peak at zero lag is sum of f[i]^2 * dx = N * dx = 1/2. The off-peak values
    are what we want to minimize in absolute value.
    
    Let's compute the full autoconvolution and take the max absolute value
    excluding the central peak.
    """
    f = signed_steps.astype(np.float64)
    # Linear convolution of f with itself, then scale by dx
    # np.convolve with mode='full' gives length 2N-1
    conv = np.convolve(f, f, mode='full') * dx
    # The peak is at index N-1 (zero lag)
    # Remove the peak and take max absolute value of the rest
    if len(conv) > 1:
        # Exclude the central peak
        peak_idx = N - 1
        if peak_idx < len(conv):
            conv_no_peak = np.concatenate([conv[:peak_idx], conv[peak_idx+1:]])
        else:
            conv_no_peak = conv
        return np.max(np.abs(conv_no_peak))
    return 0.0

def _construct_lfsr_msequence(length, poly_coefficients):
    """
    Generate an m-sequence (maximal length sequence) using a linear feedback
    shift register over GF(2).
    
    length: desired sequence length (should be 2^m - 1 for some m, but we'll
            truncate or pad to get exactly 'length' bits)
    poly_coefficients: list of feedback tap positions (0-indexed from LSB)
    
    Returns a numpy array of 0s and 1s of the given length.
    """
    # For a 400-bit sequence, we can use a primitive polynomial of degree 9
    # (2^9 - 1 = 511) and truncate to 400 bits.
    # Or we could use a degree 8 polynomial (255 bits) and repeat, but that's
    # not a true m-sequence of length 400.
    
    # Let's use a known primitive polynomial of degree 9: x^9 + x^4 + 1
    # Taps at positions 9 and 4 (or equivalently, feedback from bits 8 and 3)
    
    m = 9  # degree of polynomial
    # Primitive polynomial x^9 + x^4 + 1: taps at 9 and 4
    # In LFSR terms, the feedback is XOR of bit at position m-1 and bit at position 4-1=3
    # Let's implement a simple LFSR
    
    # Initialize LFSR state to all 1s (non-zero)
    state = [1] * m
    
    sequence = []
    # Generate 2^m - 1 bits
    for _ in range(2**m - 1):
        sequence.append(state[0])
        # Compute feedback: XOR of tapped bits
        # For polynomial x^9 + x^4 + 1, taps are at positions 9 and 4
        # In our state array (index 0 is LSB), taps are at indices 8 and 3
        feedback = state[8] ^ state[3]
        # Shift right and insert feedback at MSB
        state = [feedback] + state[:-1]
    
    # Truncate or pad to desired length
    if len(sequence) >= length:
        return np.array(sequence[:length], dtype=np.int8)
    else:
        # Pad by repeating
        reps = (length // len(sequence)) + 1
        padded = np.tile(np.array(sequence, dtype=np.int8), reps)[:length]
        return padded

def _compute_score_from_steps(signed_steps):
    """Compute the upper bound on C3 from signed step heights."""
    return _autoconvolution_upper_bound(signed_steps)

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to improve the autoconvolution upper bound.
    
    Strategy:
    1. Start with an m-sequence from an LFSR (exploits flat autocorrelation).
    2. Map bits to {-1, 1}.
    3. Optionally refine with local search to reduce off-peak autoconvolution.
    
    Returns a numpy array of 400 values in {-1, 1}.
    """
    start_time = time.time()
    
    # Generate m-sequence based signed step function
    # Use a primitive polynomial of degree 9 (511 bits), truncate to 400
    bits = _construct_lfsr_msequence(N, [8, 3])
    # Map 0 -> -1, 1 -> 1
    signed = 2 * bits - 1  # {0,1} -> {-1,1}
    signed = signed.astype(np.int8)
    
    current_score = _compute_score_from_steps(signed)
    best_signed = signed.copy()
    best_score = current_score
    
    # Local search: try flipping individual bits to reduce the score
    # Use a greedy approach with a time budget
    deadline = start_time + _DEADLINE
    
    # Precompute the autoconvolution for efficiency
    # For each flip at position i, the change in autoconvolution can be computed
    # But for simplicity, we'll just recompute if the time budget allows
    
    n_flips = 0
    improvement_threshold = 1e-12
    
    # Simple local search: iterate through positions, try flipping each
    # Multiple passes until no improvement or time runs out
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try flipping bit i
            signed[i] = -signed[i]
            new_score = _compute_score_from_steps(signed)
            
            if new_score < best_score - improvement_threshold:
                best_score = new_score
                best_signed = signed.copy()
                improved = True
                n_flips += 1
            else:
                # Revert
                signed[i] = -signed[i]
        
        if not improved:
            break
    
    return best_signed

# EVOLVE-BLOCK-END
