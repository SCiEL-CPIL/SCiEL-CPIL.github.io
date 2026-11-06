"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct the extended quadratic residue code of length 18.
    This is a linear [18, 9, 6] code with 512 codewords.
    """
    # The generator polynomial for the QR(17) code is g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients for x^0 to x^8: 1, 0, 0, 1, 1, 1, 0, 0, 1
    # As an integer (LSB is x^0): 1 + 8 + 16 + 32 + 256 = 313
    g_poly = 313
    
    # Generate the 9 basis vectors for the length-17 cyclic code
    # The basis vectors are x^i * g(x) for i = 0..8
    basis_17 = [g_poly << i for i in range(9)]
    
    # Extend to length 18 by adding a parity bit
    # The parity bit is 1 if the weight of the 17-bit vector is odd, 0 otherwise
    # We append it as the 18th bit (bit 17)
    basis_18 = []
    for cw in basis_17:
        weight = cw.bit_count()
        parity = weight % 2
        # Append parity bit at position 17
        cw_18 = cw | (parity << 17)
        basis_18.append(cw_18)
        
    # Generate all 512 codewords by taking all linear combinations of the 9 basis vectors
    code = []
    for mask in range(1 << 9):
        cw = 0
        for i in range(9):
            if (mask >> i) & 1:
                cw ^= basis_18[i]
        code.append(cw)
        
    return code

# EVOLVE-BLOCK-END
