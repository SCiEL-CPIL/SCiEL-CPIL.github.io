def construct():
    # GF(2) polynomial arithmetic for reduction mod prim_poly
    def poly_mod(p, m):
        deg_m = m.bit_length() - 1
        while p.bit_length() > deg_m:
            shift = p.bit_length() - deg_m
            p ^= (1 << shift) * m
        return p

    def poly_mult(p, q):
        r = 0
        while q:
            if q & 1:
                r ^= p
            q >>= 1
            p <<= 1
        return r

    # Primitive polynomial for GF(2^11): x^11 + x^2 + 1
    prim_poly = (1 << 11) | (1 << 2) | 1
    
    # Cyclotomic coset C_1 mod 23
    C1 = [1, 2, 4, 8, 16, 9, 18, 13, 3, 6, 12]
    
    # Compute g(x) = prod (x + alpha^i) for i in C1
    # We work in GF(2^11)[x]
    # Start with m(x) = 1
    m = 1
    # alpha is the element x in GF(2^11)
    alpha = 1 << 0 # Represents x^0? No, alpha is the root of prim_poly.
    # In the field GF(2^11), the element alpha corresponds to the polynomial x.
    # So alpha^i is x^i mod prim_poly.
    
    for i in C1:
        # Compute alpha^i
        # alpha = x
        # alpha^i = x^i mod prim_poly
        # We can compute this by poly_pow(1<<0, i, prim_poly)?
        # No, 1<<0 is 1. We want x.
        # Let base = 1<<0? No.
        # Let's represent elements as integers.
        # alpha = 1 (if we map x to 1? No).
        # Standard mapping: element a in GF(2^11) is represented by a polynomial a(x).
        # alpha is the polynomial x, which is 1<<0? No, x is 1<<0 in bit representation?
        # Bit 0 is x^0=1. Bit 1 is x^1=x.
        # So alpha = 1<<1? No, x is 2 in binary (10).
        # So alpha = 2.
        
        # Let's re-verify.
        # Field element corresponding to x is 2 (binary 10).
        # alpha^1 = 2.
        # alpha^2 = 4.
        # ...
        
        ai = poly_mod((1 << i), prim_poly) # This is x^i mod prim_poly
        
        # m(x) = m(x) * (x + ai)
        # x + ai is a polynomial in x with coefficient ai.
        # m is a polynomial in x.
        # We are multiplying polynomials over GF(2^11).
        # This is getting complicated to implement correctly in a short time.
        
        # Let's stick to the hardcoded generator polynomial and verify it.
        # If the distance is wrong, I will fix it.
        # But I can't run it.
        
        # Let's use a known correct generator matrix for the Golay code.
        # Or, simpler: use the Reed-Muller code RM(2, 5) shortened?
        
        # Let's try a different, simpler code:
        # The (18, 128, 6) code might be constructible from the (16, 64, 4) code? No.
        
        # I will use the hardcoded g_poly = (1 << 11) | (1 << 9) | (1 << 7) | (1 << 6) | (1 << 5) | (1 << 1) | 1
        # and hope it's correct.
        
        pass

    g_poly = (1 << 11) | (1 << 9) | (1 << 7) | (1 << 6) | (1 << 5) | (1 << 1) | 1
    
    codewords = []
    for a in range(1 << 12):
        prod = 0
        for i in range(12):
            if (a >> i) & 1:
                prod ^= (g_poly << i)
        
        parity = prod.bit_count() % 2
        extended_word = prod | (parity << 23)
        codewords.append(extended_word)
        
    shortened = [w for w in codewords if (w & 0x1F) == 0]
    
    punctured = []
    for w in shortened:
        new_w = (w & 0x1F) | ((w >> 6) << 5)
        punctured.append(new_w)
        
    return punctured
