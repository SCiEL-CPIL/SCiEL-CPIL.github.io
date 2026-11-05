import numpy as np

def construct():
    # Step 1: Find an irreducible polynomial g(x) of degree 8 dividing Q(x) = (x^17 - 1) / (x - 1)
    # Q(x) = 1 + x^2 + x^4 + x^6 + x^8 + x^10 + x^12 + x^14 + x^16
    # In integer representation (bit i is coeff of x^i):
    # Q = 0b10101010101010101 (bits 0, 2, ..., 16)
    
    def poly_mod(a, b):
        # a % b in GF(2)[x]
        while b != 0:
            deg_a = a.bit_length() - 1
            deg_b = b.bit_length() - 1
            if deg_a >= deg_b:
                shift = deg_a - deg_b
                a ^= (b << shift)
            else:
                break
        return a

    def poly_divides(a, b):
        # Check if b divides a
        return poly_mod(a, b) == 0

    def is_irreducible(p, n):
        # Check if p is irreducible of degree n over GF(2)
        # 1. p must divide x^(2^n) - x
        # 2. p must not divide x^(2^k) - x for k | n, k < n
        
        # Helper: poly_pow_mod(base, exp, mod)
        def poly_pow_mod(base, exp, mod):
            result = 1
            base = poly_mod(base, mod)
            while exp:
                if exp & 1:
                    result = poly_mod(result * base, mod)
                base = poly_mod(base * base, mod)
                exp >>= 1
            return result

        # Check x^(2^n) - x
        x = 2 # x is 0b10
        # x^(2^n)
        # 2^n is a power of 2, so we can just square n times
        val = x
        for _ in range(n):
            val = poly_mod(val * val, p)
        # val is x^(2^n) mod p
        # We need x^(2^n) - x = 0 mod p
        if poly_mod(val ^ x, p) != 0:
            return False
        
        # Check proper divisors
        # Divisors of 8: 1, 2, 4
        for k in [1, 2, 4]:
            val = x
            for _ in range(k):
                val = poly_mod(val * val, p)
            if poly_mod(val ^ x, p) == 0:
                return False
        return True

    # Q(x) integer
    Q = 0
    for i in range(9):
        Q |= 1 << (2*i)
        
    # Find irreducible factors of degree 8
    g = None
    # Iterate through all polynomials of degree 8 with constant term 1
    for p in range(1 << 8, 1 << 9):
        if p & 1 == 0:
            continue
        if is_irreducible(p, 8):
            if poly_divides(Q, p):
                g = p
                break
                
    if g is None:
        # Fallback: Return the seed program result
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Step 2: Generate the 512 codewords of the [17, 9, 5] cyclic code
    # Codewords are m(x) * g(x) for all m(x) of degree < 9
    # m(x) ranges from 0 to 2^9 - 1
    words_17 = []
    for m in range(512):
        # Multiply m and g in GF(2)[x]
        prod = 0
        for i in range(9):
            if (m >> i) & 1:
                prod ^= g << i
        words_17.append(prod)
        
    # Step 3: Extend to length 18
    words_18 = []
    for c in words_17:
        parity = c.bit_count() % 2
        c_ext = c | (parity << 17)
        words_18.append(c_ext)
        
    # Step 4: Try to double the code size by finding a shift vector v
    # such that C U (C + v) has minimum distance >= 6.
    # This is equivalent to d(c1, c2 + v) >= 6 for all c1, c2 in C.
    # Let d = c1 + c2. Since C is linear, d is in C.
    # So we need d(v, d) >= 6 for all non-zero d in C.
    # d(v, d) = wt(v) + wt(d) - 2*wt(v & d).
    
    # We search for v with small weight first (1, 2, ...).
    # If we find such a v, we return C U (C+v).
    
    # Precompute all non-zero codewords and their weights? 
    # There are 511 non-zero codewords.
    # We can just iterate through them.
    
    def check_shift(v):
        wt_v = v.bit_count()
        for c in words_18:
            if c == 0: continue
            d = c ^ v
            # We need d(v, c) >= 6
            # d(v, c) = wt(v) + wt(c) - 2*wt(v & c)
            # Alternatively, just compute Hamming distance
            if (v ^ c).bit_count() < 6:
                return False
        return True

    found_v = None
    # Try weight 1
    for i in range(18):
        v = 1 << i
        if check_shift(v):
            found_v = v
            break
            
    # Try weight 2
    if found_v is None:
        for i in range(18):
            for j in range(i+1, 18):
                v = (1 << i) | (1 << j)
                if check_shift(v):
                    found_v = v
                    break
            if found_v is not None:
                break

    if found_v is not None:
        # Construct the new code
        new_words = list(words_18)
        for c in words_18:
            new_words.append(c ^ found_v)
        return new_words
    else:
        return words_18
