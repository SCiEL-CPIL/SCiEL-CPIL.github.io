"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _compute_c3(h):
    """
    Compute the autoconvolution upper bound C3 for a signed step function h.
    C3 is defined as the maximum absolute value of the autocorrelation at non-zero lags,
    normalized by the zero-lag value (which is N for binary +/-1 sequences).
    Lower is better.
    """
    # Compute autocorrelation via FFT
    # Pad to at least 2N to avoid circular convolution artifacts
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0 (sum of squares = N)
    # r[1] to r[N-1] are positive lags
    # For real signals, autocorrelation is symmetric, so we only need positive lags
    
    # The zero-lag value
    r0 = r[0]
    
    # Max absolute value at non-zero lags
    # Positive lags: r[1:N]
    max_sidelobe = np.max(np.abs(r[1:N]))
    
    # C3 is the ratio (upper bound on normalized autocorrelation peak)
    return max_sidelobe / r0


def _generate_lfsr_sequence(n, poly_taps, init_state):
    """
    Generate a sequence using a linear feedback shift register.
    poly_taps: list of bit positions (0-indexed from LSB) for feedback taps.
    init_state: initial register state (non-zero).
    Returns a sequence of +1/-1 of length n.
    """
    m = 9  # degree of polynomial
    reg = init_state
    if reg == 0:
        reg = 1  # avoid all-zero state
    
    seq = []
    for i in range(n):
        seq.append(1 if (reg & 1) else -1)
        # Shift right and apply feedback
        # feedback = XOR of bits at specified tap positions
        feedback = 0
        for tap in poly_taps:
            feedback ^= (reg >> tap) & 1
        reg = (reg >> 1) | (feedback << (m - 1))
    
    return np.array(seq, dtype=np.float64)


def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_c3 = np.inf
    
    # Known primitive polynomials for degree 9 (taps are positions where feedback is taken)
    # Format: list of tap positions (0-indexed)
    primitive_polys = [
        [0, 4],      # x^9 + x^4 + 1
        [0, 5, 4, 3], # x^9 + x^5 + x^4 + x^3 + 1
        [0, 8],      # x^9 + x^8 + 1
        [0, 1, 4],   # x^9 + x + x^4 + 1
        [0, 2, 7],   # x^9 + x^2 + x^7 + 1
        [0, 6, 7],   # x^9 + x^6 + x^7 + 1
        [0, 3, 6],   # x^9 + x^3 + x^6 + 1
        [0, 1, 7],   # x^9 + x + x^7 + 1
        [0, 2, 8],   # x^9 + x^2 + x^8 + 1
        [0, 5, 7],   # x^9 + x^5 + x^7 + 1
    ]
    
    # Try multiple LFSR configurations with different polynomials and initial states
    for taps in primitive_polys:
        for init in range(1, 16):
            if time.time() - start_time > deadline - 15:
                break
            h = _generate_lfsr_sequence(N, taps, init)
            c3 = _compute_c3(h)
            if c3 < best_c3:
                best_c3 = c3
                best_h = h.copy()
        if time.time() - start_time > deadline - 15:
            break
    
    # Also try some random sequences as fallback/initialization
    for i in range(20):
        if time.time() - start_time > deadline - 10:
            break
        h = rng.choice([-1, 1], size=N).astype(np.float64)
        c3 = _compute_c3(h)
        if c3 < best_c3:
            best_c3 = c3
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_c3 = _compute_c3(best_h)
    
    # Local search refinement using coordinate descent
    h = best_h.copy()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < deadline - 5:
        # Try flipping a random position
        idx = rng.integers(0, N)
        old_val = h[idx]
        h[idx] *= -1
        
        new_c3 = _compute_c3(h)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            best_h = h.copy()
            no_improve_count = 0
        else:
            h[idx] = old_val
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Try a few random restarts
            improved = False
            for _ in range(3):
                if time.time() - start_time > deadline - 5:
                    break
                h_restart = rng.choice([-1, 1], size=N).astype(np.float64)
                c3_restart = _compute_c3(h_restart)
                if c3_restart < best_c3:
                    best_c3 = c3_restart
                    best_h = h_restart.copy()
                    h = best_h.copy()
                    no_improve_count = 0
                    improved = True
                    break
            if not improved:
                # Try LFSR restart
                taps = primitive_polys[rng.integers(0, len(primitive_polys))]
                init = rng.integers(1, 512)
                h_lfsr = _generate_lfsr_sequence(N, taps, init)
                c3_lfsr = _compute_c3(h_lfsr)
                if c3_lfsr < best_c3:
                    best_c3 = c3_lfsr
                    best_h = h_lfsr.copy()
                    h = best_h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count = 0  # Reset to avoid infinite restart loop
    
    return best_h
# EVOLVE-BLOCK-END
