"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the 512-word extended quadratic-residue code of length 18.
    # This is a linear [18, 9, 6] code, which is optimal for A_2(18, 6).
    
    # Generator polynomial for the QR code of length 17:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Basis vectors for the cyclic code are x^i * g(x) for i=0..8.
    # Each has weight 5.
    # Extend by a parity bit (bit 17) to make weight even (6).
    
    # Bits set in g(x): 0, 3, 4, 5, 8
    # g_val = 1 + 2^3 + 2^4 + 2^5 + 2^8 = 1 + 8 + 16 + 32 + 256 = 313
    
    basis = []
    g_val = 313
    for i in range(9):
        v = g_val << i
        # Check if any bit overflowed bit 16 (index 16 is the 17th bit)
        # Max shift is 8. g_val has max bit 8. 8+8=16. So max bit index is 16.
        # Fits in 17 bits (indices 0..16).
        
        # Calculate weight of v to determine parity bit
        w = v.bit_count()
        # w is always 5 for these basis vectors
        # Parity bit is 1 if w is odd
        parity = w % 2
        # Extend to 18 bits
        extended_v = v | (parity << 17)
        basis.append(extended_v)
        
    words = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
        words.append(word)
        
    return words
# EVOLVE-BLOCK-END
