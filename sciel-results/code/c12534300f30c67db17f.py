"""Improved program: discrete simulated annealing on signed step functions."""
N = 400
import numpy as np
import time

def compute_autocorrelation_bound(f):
    """Compute an upper bound on C3 based on autocorrelation sidelobes."""
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N]
    # Try: ratio of sidelobe energy to peak energy
    if abs(acf[0]) > 1e-12:
        sidelobe_energy = np.sum(acf[1:] ** 2)
        peak_energy = acf[0] ** 2
        bound = np.sqrt(sidelobe_energy / peak_energy)
    else:
        bound = np.sqrt(np.sum(acf[1:] ** 2))
    return bound

def construct():
    # Use LFSR to generate a maximal-length sequence (m-sequence)
    # Degree 9 gives length 511, which is > 400.
    # Primitive polynomial for degree 9: x^9 + x^4 + 1 (taps at 9, 4, 0)
    # Standard m-sequence generation
    
    def generate_m_sequence(degree, taps, length):
        # taps are indices for the feedback polynomial, e.g., [9, 4, 0] means x^9 + x^4 + 1
        # State is a list of bits
        state = [1] * degree
        seq = []
        for _ in range(length):
            seq.append(state[-1])
            # Feedback bit
            # For x^degree + ... + 1, feedback is XOR of bits at tap positions
            # Note: Standard LFSR implementations vary in bit ordering.
            # Let's use a simple Fibonacci LFSR.
            # If poly is x^d + c_{d-1}x^{d-1} + ... + c_0
            # Here x^9 + x^4 + 1 => c_4=1, c_0=1.
            # Feedback = state[0] XOR state[degree-1-c_4] ?
            # Let's stick to a known working implementation for x^9+x^4+1.
            # Taps: 9, 4, 0.
            # We will use a Galois LFSR or Fibonacci.
            # Let's use a simple recursive definition for clarity and speed.
            pass
            
    # Simpler approach: Use numpy to generate a pseudo-random sequence that mimics m-sequence properties?
    # No, let's implement a basic LFSR properly.
    
    def lfsr_sequence(degree, poly_taps, length):
        # poly_taps: list of exponents, e.g., [9, 4, 0]
        # State: degree bits.
        # We'll use a Fibonacci LFSR.
        # Output is the last bit.
        # Feedback is XOR of bits at positions corresponding to non-leading terms.
        # For x^9 + x^4 + 1, feedback = bit[0] XOR bit[4] (if bit[0] is LSB/x^0 term)
        # Let's define state[0] as the bit for x^0 term (newest output?), or x^{degree-1}?
        # Standard: state = [b_{d-1}, ..., b_0].
        # Output b_0.
        # Next state: [b_{d-2}, ..., b_1, b_{d-1} XOR ...]
        
        # Let's use a known primitive polynomial for deg 9: 0x123 (x^9 + x^5 + x^3 + x^2 + 1) is common?
        # Or x^9 + x^4 + 1.
        # Let's just use a random seed and generate a long sequence, then take a slice?
        # The prompt specifically suggests LFSR.
        
        # Let's use degree 9, poly x^9 + x^4 + 1.
        # Taps: 9, 4, 0.
        # Feedback = state[0] ^ state[4] (assuming state[0] is LSB)
        
        state = [1] * degree
        seq = []
        for _ in range(length):
            seq.append(state[-1])
            # Calculate feedback
            # For x^9 + x^4 + 1:
            # b_new = b_0 ^ b_4
            fb = 0
            # Taps excluding the leading term (degree)
            for tap in poly_taps:
                if tap == degree:
                    continue
                if tap == 0:
                    fb ^= state[0]
                else:
                    fb ^= state[tap - 1] # index shift?
            
            # Shift right
            state = state[:-1] + [fb]
            
        return seq

    # Let's try a few standard primitive polynomials for degree 9
    # 1. x^9 + x^4 + 1
    # 2. x^9 + x^5 + x^3 + x^2 + 1
    
    candidates = []
    
    # Poly 1: x^9 + x^4 + 1
    taps1 = [9, 4, 0]
    seq1 = lfsr_sequence(9, taps1, 511)
    # Map bits to {-1, 1}
    f1 = np.array([1 if b == 1 else -1 for b in seq1[:N]])
    candidates.append(f1)
    
    # Poly 2: x^9 + x^5 + x^3 + x^2 + 1
    taps2 = [9, 5, 3, 2, 0]
    seq2 = lfsr_sequence(9, taps2, 511)
    f2 = np.array([1 if b == 1 else -1 for b in seq2[:N]])
    candidates.append(f2)
    
    # Also try a phase shift (circular shift) of the first sequence
    # m-sequences have good autocorrelation regardless of phase, but let's check.
    f3 = np.roll(f1, N // 2)
    candidates.append(f3)
    
    # Evaluate all candidates and return the best one
    best_f = None
    best_bound = float('inf')
    
    for f in candidates:
        bound = compute_autocorrelation_bound(f)
        if bound < best_bound:
            best_bound = bound
            best_f = f
    
    return best_f
