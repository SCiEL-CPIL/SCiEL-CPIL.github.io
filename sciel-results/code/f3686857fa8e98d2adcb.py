"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_periodic(h):
    """Compute periodic autocorrelation of a ±1 sequence using FFT."""
    f = np.fft.fft(h.astype(float))
    power = f * np.conj(f)
    acorr = np.fft.ifft(power).real
    return acorr


def _generate_lfsr_sequence(seed_poly, init_state, length):
    """
    Generate a binary sequence from an LFSR.
    seed_poly: integer representing the feedback polynomial (taps)
    init_state: initial state (non-zero integer)
    length: number of bits to generate
    """
    seq = []
    state = init_state
    # Determine bit width from the polynomial
    bit_width = seed_poly.bit_length()
    
    for _ in range(length):
        # Output the least significant bit (or most, depending on convention)
        # Let's use LSB as output
        seq.append(1.0 if (state & 1) else -1.0)
        
        # Shift right
        state >>= 1
        # Feedback: XOR of bits at positions specified by the polynomial
        # The polynomial is defined by its taps. 
        # If seed_poly has a bit set at position k, it means x^k is a term.
        # Standard LFSR: new_bit = XOR(state & mask)
        # We need to determine the feedback bit.
        # Let's use a standard Fibonacci LFSR.
        # The polynomial is typically defined by the positions of the 1s in the binary representation, excluding the highest bit (which is implicit x^L).
        # Actually, for simplicity, let's just use a fixed set of taps that we know are primitive or near-primitive for small lengths, 
        # or just use random linear feedback if we don't have a specific primitive poly for length 400.
        # Since N=400 is not a power of 2 minus 1 (M-sequence length), we can't have a perfect M-sequence of length 400.
        # However, the strategy suggests using LFSR-derived sequences.
        # We will use a pseudo-random feedback based on the current state to mimic LFSR behavior, 
        # or simply use a known good primitive polynomial for a smaller length and repeat/extend?
        # The prompt says "first N bits of a linear feedback shift register (LFSR) with a primitive polynomial".
        # Since 400 is not 2^m-1, we cannot have a maximal length sequence of exactly 400.
        # We will generate a sequence of length 2^m-1 where 2^m-1 >= 400 (e.g., 2^9-1=511) and take the first 400.
        
        # Let's use a primitive polynomial for length 9 (511 bits).
        # Primitive polynomial for GF(2^9): x^9 + x^4 + 1 (0b100010001 = 273)
        # Or x^9 + x^5 + x^4 + x^2 + 1 (0b1001111001 = 633)
        # Let's use 273 (x^9 + x^4 + 1)
        
        # To do this properly, we need to maintain a state of width 9.
        # My simple shift above assumes a fixed width. Let's refactor to use a specific width.
        
        pass # This function is a placeholder for the logic below

def _generate_lfsr_sequence_proper(taps, init_state, length):
    """
    Generate a sequence using an LFSR with specified taps and state width.
    taps: list of bit positions to XOR (0-indexed from LSB)
    init_state: initial integer state
    length: number of bits to generate
    """
    seq = []
    state = init_state
    # Mask for the state width
    width = max(taps) + 1
    mask = (1 << width) - 1
    
    for _ in range(length):
        # Output LSB
        seq.append(1.0 if (state & 1) else -1.0)
        
        # Calculate feedback
        feedback = 0
        for t in taps:
            if (state >> t) & 1:
                feedback ^= 1
        
        # Shift right and insert feedback at MSB
        state = ((state >> 1) | (feedback << (width - 1))) & mask
        
    return np.array(seq)


def _local_search(h, R, deadline):
    """Perform local search with incremental autocorrelation updates."""
    best_score = np.max(np.abs(R[1:]))
    best_h = h.copy()
    h = h.copy()
    R = R.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for j in range(N):
            s = h[j]
            idx_plus = (j + np.arange(N)) % N
            idx_minus = (j - np.arange(N)) % N
            delta = -2.0 * s * (h[idx_plus] + h[idx_minus])
            new_R = R + delta
            new_score = np.max(np.abs(new_R[1:]))
            
            if new_score < best_score - 1e-12:
                h[j] = -s
                R = new_R
                best_score = new_score
                best_h = h.copy()
                improved = True
            
            if time.time() > deadline:
                break
    
    return best_h, best_score


def construct():
    deadline = time.time() + 140
    
    np.random.seed(42)
    best_h = None
    best_score = float('inf')
    
    # Strategy: Use LFSR sequences as starting points, then refine with local search.
    # We will try several different LFSR configurations (different taps/init states)
    # and also some random starts to ensure robustness.
    
    # LFSR Configuration:
    # We need a length that covers 400. Let's use a 9-bit LFSR (period 511).
    # Primitive polynomial: x^9 + x^4 + 1.
    # Taps (positions to XOR for feedback): 
    # In a Fibonacci LFSR, if the polynomial is x^9 + x^4 + 1, 
    # the feedback is usually computed from the bit that is being shifted out (LSB) 
    # XORed with the bits corresponding to the other terms? 
    # Let's use a standard implementation:
    # State is 9 bits. 
    # Feedback bit = bit[0] XOR bit[4] (if poly is x^9+x^4+1, the taps are 0 and 4? Or 4 and 8?)
    # Let's stick to a known working setup or just use random taps that likely produce long periods.
    # Actually, for the purpose of "inspiration", the key is using structured initializations.
    
    # Let's define a few LFSR-like initializations
    lfsr_starts = []
    
    # 1. Standard 9-bit LFSR with poly x^9+x^4+1 (taps 0, 4, 8)
    # Feedback = bit0 XOR bit4 XOR bit8? No, usually feedback is one bit.
    # Let's use taps [0, 4] for a 9-bit register, meaning new_bit = bit0 ^ bit4.
    # This corresponds to x^9 + x^5 + 1? 
    # Let's just use a few random "linear" patterns as seeds that are not fully random, 
    # or simply generate a 511-bit M-sequence and take the first 400.
    
    # Generate a proper M-sequence of length 511
    # Primitive poly for n=9: x^9 + x^4 + 1 (0b100010001)
    # Taps: 0, 4, 8. 
    # Implementation:
    def get_m_sequence_511(init_state=1):
        taps = [0, 4, 8]
        width = 9
        mask = (1 << width) - 1
        state = init_state & mask
        if state == 0:
            state = 1
        seq = []
        for _ in range(511):
            seq.append(1.0 if (state & 1) else -1.0)
            # Feedback: XOR of bits at taps
            fb = 0
            for t in taps:
                if (state >> t) & 1:
                    fb ^= 1
            state = ((state >> 1) | (fb << (width - 1))) & mask
        return np.array(seq[:N])

    # Add LFSR starts
    for i in range(1, 10):
        lfsr_starts.append(get_m_sequence_511(init_state=i))
    
    # Add some random starts as well
    num_random_starts = 20
    for i in range(num_random_starts):
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        lfsr_starts.append(h_rand)
    
    # Run local search on each start
    for h_init in lfsr_starts:
        if time.time() > deadline - 15:
            break
        
        R = _autocorr_periodic(h_init)
        h_opt, score = _local_search(h_init, R, deadline)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
