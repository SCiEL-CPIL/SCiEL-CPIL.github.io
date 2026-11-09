"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _generate_m_sequence(length_target):
    """Generate an m-sequence of length >= length_target using LFSR."""
    # Use a primitive polynomial for m=9 (length 511)
    # x^9 + x^4 + 1 (tap positions 9 and 4, 0-indexed from LSB)
    m = 9
    # LFSR state, 9 bits
    state = 1  # Non-zero initial state
    
    sequence = []
    total_len = (1 << m) - 1  # 511
    
    for _ in range(total_len):
        sequence.append(state & 1)
        # Compute new bit: XOR of bits at tap positions
        # For polynomial x^9 + x^4 + 1, taps at positions 0 and 4 (1-indexed: 1 and 5)
        # Standard: new_bit = (state >> 0) ^ (state >> 4)  for x^9+x^4+1
        new_bit = (state >> 0) ^ (state >> 4)
        state = (state >> 1) | (new_bit << (m - 1))
    
    # Convert 0/1 to -1/+1 (or +1/-1)
    seq = np.array(sequence, dtype=np.float64)
    seq = 2.0 * seq - 1.0  # 0 -> -1, 1 -> +1
    
    # Truncate or pad to target length
    if len(seq) >= length_target:
        return seq[:length_target]
    else:
        # Pad by repeating
        repeats = (length_target // len(seq)) + 1
        padded = np.tile(seq, repeats)
        return padded[:length_target]


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate m-sequence based signed step function
    heights = _generate_m_sequence(N)
    
    # Verify it's a valid sequence
    if len(heights) != N:
        heights = np.resize(heights, N)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen, but guard
        pass
    
    return heights
# EVOLVE-BLOCK-END
