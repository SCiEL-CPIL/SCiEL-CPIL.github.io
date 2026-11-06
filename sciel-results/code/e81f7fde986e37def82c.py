"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # GF(2^4) arithmetic
    # Primitive polynomial: x^4 + x + 1 (0b10011)
    PRIMITIVE_POLY = 0b10011
    
    def gf_mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0b10000: # x^4
                a ^= PRIMITIVE_POLY
        return result

    def gf_pow(a, p):
        result = 1
        while p:
            if p & 1:
                result = gf_mul(result, a)
            a = gf_mul(a, a)
            p >>= 1
        return result

    # Find primitive element alpha
    # alpha = 1 is not primitive (order 1)
    # Try alpha = 2 (x)
    alpha = 2
    # Verify alpha is primitive (order 15)
    # alpha^15 should be 1
    if gf_pow(alpha, 15) != 1:
        # Try other elements
        for a in range(2, 16):
            if gf_pow(a, 15) == 1 and gf_pow(a, 5) != 1 and gf_pow(a, 3) != 1:
                alpha = a
                break

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Compute generator polynomial g(x)
    # g(x) = prod (x - alpha^i) for i in Q
    # Represent polynomials as list of coefficients [c0, c1, ..., ck]
    g = [1] # 1
    
    for i in Q:
        root = gf_pow(alpha, i)
        # Multiply g(x) by (x - root)
        # (x - root) = [ -root, 1 ] = [ root, 1 ] in GF(2)
        # New g = g * (x + root)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            # g[j] * x^j * (x + root) = g[j]*x^(j+1) + g[j]*root*x^j
            new_g[j+1] ^= g[j]
            new_g[j] ^= gf_mul(g[j], root)
        g = new_g
        
    # g should have coefficients in GF(2)
    # Convert to binary integer (bitmask)
    g_int = 0
    for j in range(len(g)):
        if g[j] == 1:
            g_int |= (1 << j)
            
    # g_int represents the generator polynomial of the cyclic code of length 17
    # The code consists of all c(x) = m(x) * g(x) mod (x^17 - 1)
    # m(x) has degree < 9, so there are 2^9 = 512 messages
    
    words = []
    for m in range(512):
        # m is an integer representing the message polynomial
        # c(x) = (m * g) mod (x^17 - 1)
        # Perform polynomial multiplication
        # Since we are in GF(2), multiplication is bitwise XOR shift
        c = 0
        mm = m
        while mm:
            if mm & 1:
                c ^= g_int
            mm >>= 1
            g_int <<= 1
        
        # Reduce mod x^17 - 1
        # c might have degree >= 17
        # x^17 = 1, so x^k = x^(k mod 17)
        # We need to fold the bits
        # Simple way: iterate and fold
        c_final = 0
        for k in range(34): # Max degree is 8+8=16? No, m has deg 8, g has deg 8, product deg 16.
             # Wait, m has 9 bits (0..8), g has 9 bits (0..8). Product has degree 16.
             # So we only need to handle bits 0..16.
             # But x^17 = 1.
             # If degree < 17, no reduction needed?
             # The cyclic code is defined mod x^17 - 1.
             # If deg(c) < 17, c is already reduced.
             # Max degree of m*g is 8+8=16.
             # So no reduction is needed for the multiplication part!
             pass
        
        # So c is just the product, which has bits 0..16
        # This gives the 17-bit codeword
        # Bit j of c corresponds to x^j
        
        # Now extend to length 18 by adding parity bit
        # Parity bit = sum of bits mod 2
        weight = c.bit_count()
        parity = weight % 2
        word_18 = c | (parity << 17)
        
        words.append(word_18)
        
    return words
# EVOLVE-BLOCK-END
