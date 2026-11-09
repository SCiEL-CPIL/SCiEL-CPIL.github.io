"""Program using LFSR-generated binary sequence for signed step function to minimize C3 upper bound."""
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    
    We model the signed step function as a binary sequence derived from the first N bits
    of a linear feedback shift register (LFSR) with a primitive polynomial.
    
    Maximal-length sequences (m-sequences) have well-known optimal periodic autocorrelation
    properties: they achieve the Welch bound or are close to it, which translates to
    minimizing the autoconvolution upper bound C3.
    
    The signed step heights are +1 and -1 (mapping 0 -> -1, 1 -> +1 or similar).
    
    We use a primitive polynomial of degree m where 2^m - 1 >= N. For N=400,
    we need m such that 2^m - 1 >= 400. 2^9 - 1 = 511 >= 400, so m=9 works.
    
    A primitive polynomial of degree 9 over GF(2) is: x^9 + x^4 + 1 (or similar).
    Let's use x^9 + x^5 + x^4 + x^3 + 1 which is known to be primitive.
    Actually, let's use a well-known primitive polynomial: x^9 + x^4 + 1.
    
    The LFSR generates a maximal-length sequence of period 2^9 - 1 = 511.
    We take the first N=400 bits.
    
    Mapping: bit 0 -> -1, bit 1 -> +1 (or vice versa; the sign flip doesn't change |autocorrelation|)
    """
    # Primitive polynomial of degree 9: x^9 + x^4 + 1
    # In standard LFSR notation, the feedback taps are at positions corresponding to the polynomial.
    # Polynomial: x^9 + x^4 + 1 means taps at degrees 9, 4, 0.
    # For an LFSR of length m=9, we use taps [4, 0] (excluding the leading x^9 which determines the new bit).
    # Actually, let's be precise. A primitive polynomial of degree m defines an LFSR of length m.
    # The polynomial x^9 + x^4 + 1 corresponds to taps at positions 4 and 0 (0-indexed from LSB).
    # The new bit is XOR of bits at positions 4 and 0.
    
    m = 9
    # Taps for x^9 + x^4 + 1: positions 4 and 0
    # In a standard Fibonacci LFSR, the new bit = bit[tap1] XOR bit[tap2] ...
    # Let's use the convention where the register holds bits b[0]...b[m-1]
    # and the new bit b_new = b[4] XOR b[0] for polynomial x^9 + x^4 + 1.
    
    # Initialize LFSR state (non-zero)
    state = [1] + [0] * (m - 1)  # Start with 100000000
    
    sequence = []
    for _ in range(N):
        # Take the current bit (let's say the LSB, b[0])
        sequence.append(state[0])
        
        # Generate new bit
        # For x^9 + x^4 + 1, new bit = state[4] XOR state[0]
        new_bit = state[4] ^ state[0]
        
        # Shift register: b[i] = b[i+1], b[0] = new_bit
        # Actually, let's shift right: new bit enters at position m-1 or position 0?
        # Let's use: state[0] is the output bit.
        # new_bit = state[4] ^ state[0]
        # state = [new_bit] + state[:-1]  # This shifts left, new bit at position 0
        
        state = [new_bit] + state[:-1]
    
    # Map bits to signed step heights: 0 -> -1, 1 -> +1
    # This creates a signed step function
    heights = np.where(np.array(sequence) == 1, 1.0, -1.0)
    
    return heights

# EVOLVE-BLOCK-END
