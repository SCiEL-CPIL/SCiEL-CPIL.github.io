"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the [18, 9, 6] extended quadratic residue code.
    # This code has 512 codewords and minimum distance 6.
    
    # Generator polynomial for the [17, 9, 6] QR code:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients for x^0 to x^16
    g_coeffs = [0] * 17
    g_coeffs[0] = 1
    g_coeffs[3] = 1
    g_coeffs[4] = 1
    g_coeffs[5] = 1
    g_coeffs[8] = 1
    
    words = []
    
    # Iterate over all 2^9 = 512 possible messages
    for msg in range(512):
        # Convert message to a vector of length 17 (padded with zeros)
        m_vec = [0] * 17
        for i in range(9):
            if (msg >> i) & 1:
                m_vec[i] = 1
        
        # Perform circular convolution modulo 2 (polynomial multiplication mod x^17 - 1)
        c_vec = [0] * 17
        for i in range(17):
            if m_vec[i]:
                for j in range(17):
                    if g_coeffs[j]:
                        c_vec[(i + j) % 17] ^= 1
        
        # Calculate parity bit for extension to length 18
        parity = sum(c_vec) % 2
        
        # Form the 18-bit word
        word = 0
        for i in range(17):
            if c_vec[i]:
                word |= (1 << i)
        if parity:
            word |= (1 << 17)
            
        words.append(word)
        
    return words
# EVOLVE-BLOCK-END
