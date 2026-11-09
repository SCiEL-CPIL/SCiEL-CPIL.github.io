import numpy as np
import time

N = 400

def _compute_metric(x):
    """Compute the C3 upper bound (autoconvolution metric) for a ±1 sequence."""
    # Periodic autocorrelation
    # R(k) = sum_i x[i] * x[(i+k) % N]
    # Use FFT for efficiency
    # For periodic autocorrelation, we can use:
    # R = ifft(fft(x) * conj(fft(x))) -- but this gives circular correlation
    # Actually for real sequences, circular autocorrelation = ifft(|fft(x)|^2)
    X = np.fft.fft(x.astype(np.float64))
    R = np.real(np.fft.ifft(X * np.conj(X)))
    # R[0] = N, R[k] for k=1..N-1 are sidelobes
    # C3 upper bound: use max absolute sidelobe or L2 norm
    sidelobes = R[1:]
    # Use the L2 norm of sidelobes as the metric (merit index related)
    # Or peak sidelobe
    # Let's use peak sidelobe as it's more standard for "upper bound"
    return np.max(np.abs(sidelobes))

def _generate_m_sequence(m):
    """Generate an m-sequence of length 2^m - 1."""
    # Use a primitive polynomial for m-bit LFSR
    # Common primitive polynomials (in x^m + ... + 1 form)
    # For m=9: x^9 + x^4 + 1 (0x21 in some representations)
    # Let's use a simple implementation
    # Primitive polynomials for various m:
    # m=9: x^9 + x^4 + 1
    # We'll use a general approach
    
    # For m=9, use polynomial x^9 + x^4 + 1
    # Taps at positions 9 and 4 (0-indexed from LSB)
    # State is 9 bits, we output one bit per step
    
    if m == 9:
        # x^9 + x^4 + 1
        taps = (1 << 4)  # tap at position 4
        length = (1 << m) - 1
    elif m == 8:
        # x^8 + x^6 + x^5 + x^4 + 1
        taps = (1 << 6) | (1 << 5) | (1 << 4)
        length = (1 << m) - 1
    elif m == 7:
        # x^7 + x^3 + 1
        taps = (1 << 3)
        length = (1 << m) - 1
    elif m == 6:
        # x^6 + x^5 + 1
        taps = (1 << 5)
        length = (1 << m) - 1
    elif m == 5:
        # x^5 + x^2 + 1
        taps = (1 << 2)
        length = (1 << m) - 1
    elif m == 4:
        # x^4 + x^3 + 1
        taps = (1 << 3)
        length = (1 << m) - 1
    else:
        # Fallback to m=9
        m = 9
        taps = (1 << 4)
        length = (1 << m) - 1
    
    # Initialize state to all ones
    state = (1 << m) - 1
    seq = []
    for i in range(length):
        # Output bit is LSB
        bit = state & 1
        seq.append(1 if bit == 1 else -1)
        # Compute next state
        # Feedback: XOR of bits at tap positions
        fb = 0
        for t in range(m):
            if (taps >> t) & 1:
                fb ^= (state >> t) & 1
        # Shift right and insert feedback at MSB
        state = (state >> 1) | (fb << (m - 1))
    
    return np.array(seq, dtype=np.float64)

def construct():
    """Construct a signed step function (sequence of ±1) to minimize C3 upper bound."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with m-sequence based initialization
    # m=9 gives length 511, truncate to 400
    mseq = _generate_m_sequence(9)
    x = mseq[:N].copy()
    
    # Evaluate initial metric
    best_x = x.copy()
    best_metric = _compute_metric(x)
    
    # Local search: single bit flips
    rng = np.random.default_rng(42)
    
    # Track elapsed time
    start_time = time.time()
    
    iterations = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random position to flip
        pos = rng.integers(0, N)
        x[pos] *= -1  # Flip the sign
        
        metric = _compute_metric(x)
        iterations += 1
        
        if metric < best_metric:
            best_metric = metric
            best_x = x.copy()
            no_improve = 0
        else:
            # Revert the flip
            x[pos] *= -1
            no_improve += 1
        
        # Periodic check
        if iterations % 1000 == 0 and time.time() > deadline - 10:
            break
    
    return best_x
