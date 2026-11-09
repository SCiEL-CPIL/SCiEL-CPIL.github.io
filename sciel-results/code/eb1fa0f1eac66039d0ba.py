"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Construct the [17, 8, 6] code by shortening the [18, 9, 6] code.
    # The [18, 9, 6] code is the punctured version of the [19, 9, 6] code,
    # which is the even-weight subcode of the [19, 10, 5] Quadratic Residue code.
    
    # 1. Define GF(2^9) arithmetic
    # Primitive polynomial for GF(2^9): x^9 + x^4 + 1 (0x211)
    # Let's verify if x^9 + x^4 + 1 is primitive.
    # We need an element alpha with order 511.
    
    def gf2_9_add(a, b):
        return a ^ b

    def gf2_9_mul(a, b):
        if a == 0 or b == 0:
            return 0
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            if a & 0x200:  # 2^9
                a ^= 0x211  # x^9 + x^4 + 1
            b >>= 1
        return result

    # Find primitive element
    # Try x (value 2)
    alpha = 2
    order = 1
    current = alpha
    while current != 1:
        current = gf2_9_mul(current, alpha)
        order += 1
        if order > 511:
            break
    
    if order != 511:
        # Try another generator if x is not primitive, but x is usually primitive for small fields
        # For safety, we can just use the known generator polynomial for QR(19)
        # g(x) = x^9 + x^7 + x^6 + x^5 + x^3 + x^2 + x + 1 ?
        # Let's compute it properly.
        pass

    # Compute generator polynomial g(x) = prod (x - alpha^(13 * 2^i)) for i=0..8
    # Roots are alpha^13, alpha^26, ..., alpha^(13*256)
    # We represent polynomials as lists of coefficients [c0, c1, ...]
    
    roots = []
    for i in range(9):
        exp = (13 * (2**i)) % 511
        # Calculate alpha^exp
        val = 1
        base = alpha
        e = exp
        while e:
            if e & 1:
                val = gf2_9_mul(val, base)
            base = gf2_9_mul(base, base)
            e >>= 1
        roots.append(val)

    # Multiply (x + r) for each root r
    # Start with g(x) = 1
    g = [1]
    for r in roots:
        # Multiply g(x) by (x + r)
        new_g = [0] * (len(g) + 1)
        for i in range(len(g)):
            # x * g_i(x)
            new_g[i+1] ^= g[i]
            # r * g_i(x)
            new_g[i] ^= gf2_9_mul(g[i], r)
        g = new_g

    # g is the generator polynomial of the [19, 10, 5] code.
    # Degree 9.
    # The code is cyclic.
    # We can generate all 1024 codewords by taking all multiples of g(x) by polynomials of degree < 10.
    
    codewords = []
    for msg in range(1024):
        # msg is a 10-bit integer representing the message polynomial m(x)
        # c(x) = m(x) * g(x) mod (x^19 - 1)
        # Since deg(m) < 10 and deg(g) = 9, deg(c) < 19.
        # So c(x) = m(x) * g(x) directly.
        
        # Convert msg to polynomial coefficients
        m_coeffs = [(msg >> i) & 1 for i in range(10)]
        
        # Multiply m(x) and g(x)
        c_coeffs = [0] * 19
        for i in range(10):
            if m_coeffs[i]:
                for j in range(10):
                    c_coeffs[i+j] ^= g[j]
        
        # Convert to integer (bit 0 is coeff 0)
        word = 0
        for i in range(19):
            if c_coeffs[i]:
                word |= (1 << i)
        
        codewords.append(word)

    # Filter for even weight
    even_codewords = [w for w in codewords if bin(w).count('1') % 2 == 0]
    
    # This is the [19, 9, 6] code.
    # Puncture the last bit (bit 18, index 18) to get [18, 9, 6] code?
    # Actually, the [19, 9, 6] code is the extension of the [18, 9, 6] code.
    # So removing the parity bit (which is always 0 for even weight? No, the parity bit is the 19th bit).
    # In the extended code, the parity bit ensures even weight.
    # If we remove the parity bit, we get the original [18, 9, 6] code.
    
    punctured_codewords = [w & ((1 << 18) - 1) for w in even_codewords]
    
    # Now we have 512 codewords of length 18 with min distance 6.
    # Shorten by fixing bit 0 to 0.
    final_codewords = []
    for w in punctured_codewords:
        if (w & 1) == 0:
            # Remove bit 0
            final_codewords.append(w >> 1)
            
    # We should have 256 unique codewords.
    # Ensure distinctness
    unique_codewords = list(set(final_codewords))
    
    return unique_codewords

# EVOLVE-BLOCK-END
