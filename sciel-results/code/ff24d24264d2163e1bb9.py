def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0

    # GF(2^8) arithmetic with primitive polynomial 0x11D (x^8 + x^4 + x^3 + x + 1)
    PRIM_POLY = 0x11D

    def gf2_mul(a, b):
        """Multiply two elements of GF(2^8)."""
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return result

    # Find an element of order 17 in GF(2^8)
    # alpha = 2 is a primitive element (order 255)
    alpha = 2
    # omega = alpha^(255/17) = alpha^15
    omega = 1
    for _ in range(15):
        omega = gf2_mul(omega, alpha)
    
    # Verify omega^17 == 1
    check = omega
    for _ in range(16):
        check = gf2_mul(check, omega)
    if check != 1:
        # Fallback: return a small valid code if something goes wrong
        # But this should not happen
        return [0, (1 << 17) - 1] 

    # Compute generator polynomial g(x) for QR(17)
    # Roots are omega^i for i in quadratic non-residues mod 17
    # Non-residues: {3, 5, 6, 7, 10, 11, 12, 14}
    # These form a single cyclotomic coset mod 17 with respect to 2:
    # 3 -> 6 -> 12 -> 7 -> 14 -> 11 -> 5 -> 10 -> 3
    # So we multiply by (x + omega^(3*2^j)) for j=0..7
    
    # Polynomial representation: list of coefficients [c0, c1, ..., cn]
    # c0 + c1*x + ... + cn*x^n
    # Coefficients are in GF(2^8), but expected to be in GF(2) at the end
    
    def poly_mul(p, q):
        """Multiply two polynomials over GF(2^8)."""
        n = len(p)
        m = len(q)
        r = [0] * (n + m - 1)
        for i in range(n):
            if p[i] != 0:
                for j in range(m):
                    if q[j] != 0:
                        r[i+j] ^= gf2_mul(p[i], q[j])
        return r

    # Start with P(x) = 1
    P = [1]
    
    # Roots exponents: 3 * 2^j mod 17 for j=0..7
    exps = []
    e = 3
    for j in range(8):
        exps.append(e)
        e = (e * 2) % 17
        
    for exp in exps:
        # Factor is (x + omega^exp) = [omega^exp, 1]
        root_val = 1
        for _ in range(exp):
            root_val = gf2_mul(root_val, omega)
        factor = [root_val, 1]
        P = poly_mul(P, factor)
        
    # P should now be the generator polynomial g(x) of degree 8
    # Its coefficients should be 0 or 1
    g_coeffs = P
    if len(g_coeffs) != 9:
        # Padding or truncation if needed, but degree should be exactly 8
        # If leading zeros, trim? No, degree is 8.
        pass
        
    # Verify coefficients are in GF(2)
    for c in g_coeffs:
        if c not in (0, 1):
            # This would be an error in construction
            break
            
    # Generate all 512 codewords
    words = []
    seen = set()
    
    # g_coeffs[i] is coefficient of x^i
    # m(x) has degree < 9, so m_coeffs[i] for i=0..8
    # c(x) = m(x) * g(x)
    
    for m_val in range(512):
        # m_coeffs
        m_coeffs = [(m_val >> i) & 1 for i in range(9)]
        
        # Convolution c = m * g
        c_coeffs = [0] * 17
        for i in range(9):
            if m_coeffs[i]:
                for j in range(9):
                    c_coeffs[i+j] ^= g_coeffs[j]
                    
        # Construct 17-bit integer
        word_17 = 0
        for k in range(17):
            if c_coeffs[k]:
                word_17 |= (1 << k)
                
        # Parity bit
        parity = word_17.bit_count() % 2
        word_18 = word_17 | (parity << 17)
        
        if word_18 not in seen:
            seen.add(word_18)
            words.append(word_18)
            
    if time.time() - start_time > DEADLINE:
        return words
        
    # Optional: Quick verification
    # n = len(words)
    # min_d = 18
    # for i in range(min(n, 20)):
    #     for j in range(i+1, min(n, 20)):
    #         d = (words[i] ^ words[j]).bit_count()
    #         if d < min_d:
    #             min_d = d
    #     if min_d < 6:
    #         break
    # if min_d < 6:
    #     # Fallback
    #     pass

    return words
