"""
Strategy: Structured Initialization via M-Sequence + Local Refinement
Instead of random starts, we generate a Maximum Length Sequence (m-sequence) 
which has near-optimal autocorrelation properties (sidelobes = -1). 
This provides a very high starting score. We then apply a limited local search 
to see if we can improve upon the theoretical limit or find a better non-periodic 
variant.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_c3(h):
    """
    Compute the autoconvolution upper bound C3 for a signed step function h.
    C3 is defined as the maximum absolute value of the autocorrelation at non-zero lags,
    normalized by the zero-lag value.
    Lower is better.
    """
    # Compute autocorrelation via FFT
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    r0 = r[0]
    if r0 == 0:
        return float('inf')
        
    # Max absolute value at non-zero lags
    # Positive lags: r[1:N]
    max_sidelobe = np.max(np.abs(r[1:N]))
    
    return max_sidelobe / r0

def _generate_m_sequence(n, taps, initial_state=1):
    """
    Generate a binary m-sequence of length 2^n - 1.
    Since N=400 is not of the form 2^n - 1, we will generate a longer m-sequence
    and take the first N bits, or pad/truncate appropriately.
    Actually, m-sequences have length 2^n - 1.
    2^9 - 1 = 511.
    We can generate a 511-length m-sequence and take the first 400 bits.
    The autocorrelation of the first 400 bits will still be very good.
    """
    # We need a prime n such that 2^n - 1 >= N.
    # N=400. 2^9 - 1 = 511.
    # Let's use n=9.
    # Taps for n=9: (9, 5) is a primitive polynomial x^9 + x^5 + 1.
    # Or (9, 4) x^9 + x^4 + 1.
    # Let's try a few known primitive polynomials for degree 9.
    # One common one is taps (9, 5) meaning feedback from bits 9 and 5 (0-indexed from LSB? or MSB?)
    # Standard LFSR: new_bit = bit[i1] XOR bit[i2] ...
    
    # Let's implement a generic LFSR generator
    # State is a list of bits
    # We want length 511
    seq_len = 511
    state = [0] * n
    state[0] = initial_state
    seq = []
    
    for _ in range(seq_len):
        # Generate next bit
        # Taps are indices in the state (0 is LSB/oldest? usually 0 is newest in some conventions, let's stick to a standard)
        # Let's use the convention: state[0] is the leftmost (oldest) or rightmost?
        # Let's use right-shift: new bit enters at the right.
        # Feedback is XOR of bits at specific positions.
        # For n=9, polynomial x^9 + x^5 + 1 corresponds to taps at 9 and 5.
        # In a 9-bit register, indices 0..8.
        # Tap 9 means the bit that is about to shift out? No, usually x^n + x^k + 1 means feedback from bit n-1 and bit k-1?
        # Let's just try a few tap combinations and check the autocorrelation.
        
        # Simpler: Just generate random sequences but seeded by a hash of the tap config?
        # No, let's implement the LFSR properly.
        
        # Let's use a known primitive polynomial for degree 9: x^9 + x^5 + 1
        # Taps: 9, 5.
        # In a shift register of size 9, if we shift right, the new bit is XOR of bit 8 and bit 4?
        # Let's assume 0-indexed from LSB (rightmost).
        # x^9 + x^5 + 1 => taps at 8 and 4 (if 1-indexed 9 and 5 -> 0-indexed 8 and 4).
        
        # Let's try taps = (8, 4)
        # new_bit = state[8] ^ state[4]
        
        # Wait, N=400. We can just take the first 400 bits of a 511-bit m-sequence.
        
        # Let's implement a flexible LFSR
        # We will try multiple tap configurations for n=9
        pass
    
    # Since implementing a robust LFSR with correct tap conventions is error-prone in a short script,
    # and we have NumPy, let's use a different structured approach:
    # Use a Barker-like sequence or just a known good pattern?
    # Actually, the "inspiration 2" mentioned `generate_m_sequence`.
    # Let's try to construct a sequence that approximates an m-sequence.
    
    # Alternative: Use a random sequence but with a specific property?
    # No, the key is the M-SEQUENCE.
    
    # Let's hardcode a 511-bit m-sequence generator.
    # Primitive polynomial for n=9: x^9 + x^5 + 1
    # Taps (0-indexed from LSB): 4 and 8?
    # Let's test a few combinations of taps for n=9 and pick the one with best autocorrelation for the first 400 bits.
    
    best_seq = None
    best_c3 = float('inf')
    
    # Possible tap pairs for n=9 (excluding trivial ones)
    # A primitive polynomial of degree n has taps at n and some k.
    # We iterate over k.
    for k in range(1, n):
        # Taps are n-1 and k-1 (0-indexed)
        # new_bit = state[n-1] ^ state[k-1]
        state = [0] * n
        state[0] = 1
        seq = []
        
        for _ in range(seq_len):
            # Shift right
            # new_bit is XOR of bits at index n-1 and k-1
            new_bit = state[n-1] ^ state[k-1]
            
            # Shift state right
            state = [new_bit] + state[:-1]
            
            # Append the bit that shifted out? Or the new bit?
            # Usually, the output is the bit that shifts out (LSB) or the new bit.
            # Let's append the new bit (MSB side?)
            # Let's just collect the state bits.
            seq.append(state[-1]) # LSB
            
        # Convert to +/- 1
        h = np.array([1 if b == 1 else -1 for b in seq])
        
        # Take first N bits
        h = h[:N]
        
        c3 = _compute_c3(h)
        if c3 < best_c3:
            best_c3 = c3
            best_seq = h.copy()
            
    return best_seq, best_c3

def _local_search(h, start_time, deadline):
    """
    Perform a limited local search to improve the sequence.
    Since we start with a very good sequence, we only need to check a few flips.
    """
    current_c3 = _compute_c3(h)
    improved = True
    
    # To speed up, we can check flips in batches or just do a few passes.
    # With N=400, one pass is 400 FFTs.
    # 400 * (FFT time ~ 10-20us) = ~10ms per pass.
    # We can do many passes.
    
    while improved and (time.time() - start_time) < deadline:
        improved = False
        best_idx = -1
        best_c3 = current_c3
        
        # Shuffle indices to avoid bias? Or just iterate.
        # Let's iterate.
        for idx in range(N):
            if (time.time() - start_time) > deadline - 2:
                break
            
            h[idx] *= -1
            c3_trial = _compute_c3(h)
            h[idx] *= -1  # revert
            
            if c3_trial < best_c3:
                best_c3 = c3_trial
                best_idx = idx
        
        if best_idx >= 0:
            h[best_idx] *= -1
            current_c3 = best_c3
            improved = True
            
    return h, current_c3

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # 1. Generate best m-sequence variant
    m_seq, m_c3 = _generate_m_sequence(9, None, 1)
    
    best_h = m_seq
    best_c3 = m_c3
    
    # 2. Try a few random permutations or phase shifts of the m-sequence?
    # M-sequences are cyclic shifts of each other.
    # Let's try a few random starting points as well, just in case the m-sequence is a local optimum trap.
    # But m-sequences are globally optimal for periodic autocorrelation.
    # For non-periodic (truncated), it's still very good.
    
    # Let's run local search on the m-sequence
    h_opt, c3_opt = _local_search(best_h.copy(), start_time, deadline)
    
    if c3_opt < best_c3:
        best_h = h_opt
        best_c3 = c3_opt
        
    # If we have time, try one random start
    if (time.time() - start_time) < deadline - 10:
        rng = np.random.default_rng(123)
        h_rand = rng.choice([-1, 1], size=N).astype(np.float64)
        h_rand_opt, c3_rand_opt = _local_search(h_rand, start_time, deadline)
        if c3_rand_opt < best_c3:
            best_h = h_rand_opt
            best_c3 = c3_rand_opt
            
    return best_h
# EVOLVE-BLOCK-END
