"""Rewritten program: LFSR-based signed step function with local refinement."""
import time
import numpy as np

N = 400
DEADLINE = 115.0  # safety margin before 120s limit

def _autoconv_bound(x):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    The metric: we compute the circular autoconvolution of x with itself,
    and the bound is the maximum off-center absolute value divided by the center value.
    Lower is better.
    """
    # Use the signed metric (standard for m-sequences)
    n = len(x)
    fft_x = np.fft.fft(x)
    conv = np.fft.ifft(fft_x * fft_x).real
    # Normalize
    center = conv[0]
    if center <= 0:
        return 1.0
    # Off-center maximum absolute value
    off_vals = np.concatenate([conv[1:], conv[:-1]])
    off_max = np.max(np.abs(off_vals))
    return off_max / center


def _lfsr_sequence(n, poly_bits, init):
    """Generate n bits from an LFSR with given polynomial and initial state."""
    seq = []
    state = init
    deg = poly_bits.bit_length() - 1
    for _ in range(n):
        seq.append(state & 1)
        # Feedback: XOR of bits at positions defined by polynomial
        feedback = 0
        for i in range(deg + 1):
            if (poly_bits >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << deg)
    return seq


def _refine(x, max_time):
    """Local search: flip individual bits to reduce the autoconvolution bound."""
    x = x.copy()
    best_val = _autoconv_bound(x)
    start = time.time()
    
    # Precompute FFT for faster evaluation
    n = len(x)
    
    # Greedy coordinate descent with multiple passes
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        for i in range(n):
            if (time.time() - start) > max_time:
                break
            x[i] *= -1
            val = _autoconv_bound(x)
            if val < best_val - 1e-12:
                best_val = val
                improved = True
            else:
                x[i] *= -1
    return x, best_val


def construct():
    t_start = time.time()
    remaining = DEADLINE - (t_start - t_start)
    
    # Try several LFSR configurations and pick the best
    # Primitive polynomials for various degrees
    # We need sequences of length 400, so use a degree >= 9 (2^9-1=511 > 400)
    
    best_x = None
    best_val = float('inf')
    
    # Candidate LFSR configurations (polynomial in binary, initial state)
    # Degree 9 primitive polynomials: x^9+x^4+1, x^9+x^5+1, x^9+x^6+x^5+x^3+1
    # In bit representation (bit i = coefficient of x^i)
    configs = [
        (0b100010001, 0x1FF),  # x^9+x^4+1, all-ones init
        (0b100100001, 0x1FF),  # x^9+x^5+1
        (0b101101001, 0x1FF),  # x^9+x^6+x^5+x^3+1
        (0b100010001, 0x0FF),  # different init
        (0b100100001, 0x0FF),
        (0b101101001, 0x0FF),
        (0b100010001, 0x123),
        (0b100100001, 0x123),
        (0b101101001, 0x123),
        # Degree 10: x^10+x^3+1, x^10+x^7+x^5+x^3+1
        (0b1000001001, 0x3FF),
        (0b1101010001, 0x3FF),
        (0b1000001001, 0x1FF),
        (0b1101010001, 0x1FF),
        # Degree 12 for longer period
        (0b1000110000101, 0xFFF),  # x^12+x^6+x^4+x+1 (need to verify primitivity)
        (0b1000000000011, 0xFFF),  # x^12+x+1
    ]
    
    for poly, init in configs:
        if time.time() - t_start > DEADLINE - 30:
            break
        bits = _lfsr_sequence(N, poly, init)
        # Convert bits to signed values: 0 -> -1, 1 -> +1
        x = np.array([1.0 if b == 1 else -1.0 for b in bits])
        val = _autoconv_bound(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Also try random initialization as fallback/comparison
    if time.time() - t_start < DEADLINE - 30:
        rng = np.random.RandomState(42)
        for seed in range(20):
            if time.time() - t_start > DEADLINE - 30:
                break
            rng = np.random.RandomState(seed)
            x = rng.choice([-1.0, 1.0], size=N)
            val = _autoconv_bound(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    
    if best_x is None:
        return np.ones(N)
    
    # Local refinement with remaining time
    remaining_time = DEADLINE - (time.time() - t_start) - 5  # 5s safety margin
    if remaining_time > 1.0:
        best_x, best_val = _refine(best_x, remaining_time)
    
    return best_x
