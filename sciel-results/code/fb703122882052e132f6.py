def construct():
        # GF(2^8) setup
        poly = 0x11D
        def gf_mult(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                if a & 0x100:
                    a ^= poly
            return res

        def gf_pow(a, n):
            res = 1
            while n:
                if n & 1:
                    res = gf_mult(res, a)
                a = gf_mult(a, a)
                n >>= 1
            return res

        # Alpha is 2
        alpha = 2
        beta = gf_pow(alpha, 15)
        
        # Roots: beta^r for r in {1, 2, 4, 8, 16, 15, 13, 9}
        exponents = [1, 2, 4, 8, 16, 15, 13, 9]
        roots = [gf_pow(beta, e) for e in exponents]
        
        # Compute generator polynomial g(x)
        # g(x) = prod (x + r_i)
        # Represent poly as list of coeffs, index = degree
        g = [1]
        for r in roots:
            # Multiply by (x + r)
            # New poly length increases by 1
            new_g = [0] * (len(g) + 1)
            for i in range(len(g)):
                # x * g_i -> degree i+1
                new_g[i+1] ^= g[i]
                # r * g_i -> degree i
                new_g[i] ^= gf_mult(r, g[i])
            g = new_g
        
        # g should be [g0, g1, ..., g8]
        # Verify g[i] is 0 or 1
        # (Optional check, but good for debugging)
        
        words = []
        for m in range(512):
            # m is 9-bit message
            # c(x) = m(x) * g(x)
            # m(x) coeffs
            m_coeffs = [(m >> i) & 1 for i in range(9)]
            
            # Convolution
            c_coeffs = [0] * 17
            for i in range(9):
                if m_coeffs[i]:
                    for j in range(9):
                        c_coeffs[i+j] ^= g[j]
            
            # Convert to integer
            word = 0
            for k in range(17):
                if c_coeffs[k]:
                    word |= (1 << k)
            
            # Parity bit
            parity = word.bit_count() % 2
            word |= (parity << 17)
            
            words.append(word)
            
        return words
