import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function with 400 intervals.
    Uses a truncated m-sequence (maximum length sequence) of length 511
    (degree 9 LFSR) to minimize autocorrelation sidelobes.
    M-sequences have two-level cyclic autocorrelation and low aperiodic
    autocorrelation, providing a better upper bound on C3 than random signs.
    """
    # Generate m-sequence of length 511 (2^9 - 1)
    # Primitive polynomial: x^9 + x^4 + 1 (taps at 9, 4, 0)
    # State is 9 bits.
    # We will generate 511 bits.
    
    length = 511
    # Initialize state with 1s (non-zero)
    state = [1] * 9
    
    seq = []
    for i in range(length):
        # Output the last bit (or first, depending on convention, doesn't matter for autocorrelation magnitude if we are consistent)
        # Let's output state[0]
        bit = state[0]
        seq.append(bit)
        
        # Shift and compute new bit
        # s_new = s[4] XOR s[0]  (from x^9 + x^4 + 1)
        # Wait, standard LFSR feedback:
        # If polynomial is x^9 + x^4 + 1, the feedback is from bit 4 and bit 0?
        # Usually, x^n + sum a_i x^i + 1.
        # Let's use a standard generator.
        # Feedback = state[4] ^ state[0]
        # New state: [state[1], state[2], ..., state[8], feedback]
        
        feedback = state[4] ^ state[0]
        state = state[1:] + [feedback]
        
    # Convert to signs
    signs = np.array([2.0 * b - 1.0 for b in seq])
    
    # Truncate to N
    return signs[:N]
# EVOLVE-BLOCK-END
