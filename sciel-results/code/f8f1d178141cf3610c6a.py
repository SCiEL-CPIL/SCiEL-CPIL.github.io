import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Approach: Constructive Deterministic Design using m-sequences (Maximum Length Sequences)
    # m-sequences have optimal autocorrelation properties for binary sequences,
    # specifically having a "three-valued" autocorrelation function where non-zero lags
    # have constant small magnitude. This minimizes the sum of cubes of autocorrelations
    # by distributing energy evenly across lags, avoiding peaks.
    
    # We need a sequence of length >= 400.
    # An m-sequence of degree m has length 2^m - 1.
    # 2^8 - 1 = 255 (too short)
    # 2^9 - 1 = 511 (sufficient)
    # We will generate a 511-bit m-sequence and take the first 400 bits.
    
    m = 9
    # Primitive polynomial for GF(2) of degree 9: x^9 + x^4 + 1
    # Coefficients for taps (excluding the leading term which is implicit in the shift):
    # The recurrence is s[n] = s[n-m] + sum(tap[i]*s[n-m+i])
    # For x^9 + x^4 + 1, the taps are at positions corresponding to x^4 and x^0.
    # In a standard LFSR implementation where we shift left and feed back:
    # feedback = s[n-9] ^ s[n-5] (if we index 0..8 in the register)
    # Let's implement a simple LFSR.
    
    # Initialize state with all 1s (non-zero initial state)
    state = [1] * m
    
    # Taps for x^9 + x^4 + 1
    # The polynomial is P(x) = x^9 + x^4 + 1.
    # The recurrence relation for the sequence s_n is:
    # s_{n+9} = s_{n+4} + s_n (mod 2)
    # So, the new bit is XOR of the bit at index 4 and the bit at index 0 of the previous state window?
    # Let's trace carefully.
    # Register: [s_n, s_{n+1}, ..., s_{n+m-1}]
    # Next bit s_{n+m} = sum( coeff[i] * s_{n+i} ) where coeff corresponds to polynomial terms.
    # P(x) = x^m + c_{m-1}x^{m-1} + ... + c_0
    # Here P(x) = x^9 + 0x^8 + ... + 0x^5 + 1x^4 + 0x^3 + 0x^2 + 0x + 1
    # So c_4 = 1, c_0 = 1.
    # s_{n+9} = c_4 * s_{n+4} + c_0 * s_n
    # s_{n+9} = s_{n+4} + s_n
    
    seq = []
    for _ in range(N + m): # Generate enough to fill N
        # Calculate feedback
        # s_{current} is at index 0 of state? 
        # Let's define state[0] as the oldest bit, state[m-1] as the newest.
        # We want to append the next bit.
        # next_bit = state[4] ^ state[0]  (indices 0-based, corresponding to s_n and s_{n+4})
        # Wait, if state is [s_n, s_{n+1}, ..., s_{n+8}], then:
        # s_{n+9} = s_{n+4} ^ s_n
        # s_n is state[0]
        # s_{n+4} is state[4]
        
        next_bit = state[4] ^ state[0]
        seq.append(state[0]) # Append the oldest bit to the output sequence
        
        # Shift state: remove oldest, add newest
        state.pop(0)
        state.append(next_bit)
        
    # We have generated N+m bits, we only need the first N
    # The sequence generated is binary (0 or 1).
    # Map to signed step function: 0 -> -1, 1 -> 1
    x = np.array(seq[:N], dtype=int)
    x = 2 * x - 1
    
    return x
# EVOLVE-BLOCK-END
