import numpy as np

def construct():
        # 1. GF(2^8) setup
        # Primitive poly: x^8 + x^4 + x^3 + x^2 + 1
        # Represented as 0b100111001 (257)
        PRIM = 0b100111001
        
        def gf_mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                if a & 0x100: # overflow bit
                    a ^= PRIM
            return res

        # Find alpha = beta^15
        # beta is a root of PRIM, so beta is represented by x.
        # We need to compute x^15 mod PRIM.
        # Actually, we can just compute powers of alpha.
        # alpha = gf_pow(x, 15)
        
        def gf_pow(base, exp):
            res = 1
            while exp:
                if exp & 1:
                    res = gf_mul(res, base)
                base = gf_mul(base, base)
                exp >>= 1
            return res

        # x is 2 (binary 10)
        alpha = gf_pow(2, 15)
        
        # Compute generator polynomial g(x)
        # Roots: alpha^i for i in QR
        QR = [1, 2, 4, 8, 9, 13, 15, 16]
        
        # g starts as 1
        g = 1 # constant 1
        
        # We need to multiply g by (x + alpha^i)
        # (x + c) * g(x)
        # If g(x) = sum g_k x^k
        # New coeff at k: g_{k-1} + c * g_k
        # This is easier to do with a list of coeffs.
        
        # Let's use a list of GF(2^8) elements for coefficients.
        g_coeffs = [1] # [1]
        
        for i in QR:
            root = gf_pow(alpha, i)
            # Multiply by (x + root)
            # new_g[k] = old_g[k-1] + root * old_g[k]
            new_g = [0] * (len(g_coeffs) + 1)
            for k in range(len(g_coeffs)):
                # Contribution to new_g[k] from old_g[k-1]
                if k > 0:
                    new_g[k] ^= g_coeffs[k-1]
                # Contribution to new_g[k] from root * old_g[k]
                new_g[k] ^= gf_mul(root, g_coeffs[k])
                # Contribution to new_g[k+1] from old_g[k]
                # Wait, x * old_g[k] goes to k+1.
                # So new_g[k+1] ^= g_coeffs[k]
            # Correct logic:
            # (x + c) * (sum a_k x^k) = sum a_k x^{k+1} + sum c a_k x^k
            # Coeff at j: a_{j-1} + c a_j
            # Max degree increases by 1.
            
            # Let's re-implement carefully
            new_g = [0] * (len(g_coeffs) + 1)
            for k in range(len(g_coeffs)):
                # a_k x^{k+1}
                new_g[k+1] ^= g_coeffs[k]
                # c a_k x^k
                new_g[k] ^= gf_mul(root, g_coeffs[k])
            g_coeffs = new_g

        # g_coeffs should be [g_0, ..., g_8]
        # They should be 0 or 1.
        # Convert to binary matrix
        # g(x) = g_0 + g_1 x + ... + g_8 x^8
        
        # Generate G_17 (9x17)
        # Row i is x^i g(x) mod (x^17 - 1)
        # Since g has degree 8, x^i g has degree 8+i.
        # For i=0..8, max degree is 16. No wrap around needed for i=0..8?
        # x^8 * g_8 x^8 = g_8 x^16. Degree 16. Fits in 17 bits.
        
        G17 = np.zeros((9, 17), dtype=np.uint8)
        for i in range(9):
            # Shift g_coeffs by i
            # g_coeffs is length 9.
            # We want the bits of x^i * g(x).
            # This is just g_coeffs shifted left by i.
            # Since len(g_coeffs)+i <= 9+8=17, it fits perfectly.
            for k in range(9):
                if g_coeffs[k] == 1:
                    G17[i, k+i] = 1
                elif g_coeffs[k] == 0:
                    pass
                else:
                    # Should not happen for binary code
                    pass

        # Extend to [18,9,6]
        # Add parity bit
        G18 = np.zeros((9, 18), dtype=np.uint8)
        G18[:, :17] = G17
        for i in range(9):
            parity = np.sum(G17[i]) % 2
            G18[i, 17] = parity

        # Generate all codewords of [18,9,6]
        codewords_18 = []
        for m in range(512):
            # m is 9-bit integer
            # c = m * G18 (over GF(2))
            c = 0
            for i in range(9):
                if (m >> i) & 1:
                    c ^= int(G18[i].sum() * 0) # No, we need to XOR the row vector
                    # Let's just compute the integer representation
            # Better:
            c_int = 0
            for i in range(9):
                if (m >> i) & 1:
                    # Convert G18[i] to int
                    row_int = 0
                    for j in range(18):
                        if G18[i, j]:
                            row_int |= (1 << j)
                    c_int ^= row_int
            codewords_18.append(c_int)

        # Shorten: keep only those with bit 17 == 0
        # Extract bits 0..16
        shortened_codes = []
        seen = set()
        for c in codewords_18:
            if (c >> 17) & 1 == 0:
                short_c = c & ((1 << 17) - 1)
                if short_c not in seen:
                    seen.add(short_c)
                    shortened_codes.append(short_c)
        
        # Now greedy search
        current_words = shortened_codes
        word_set = set(current_words)
        
        for candidate in range(1 << 17):
            if candidate in word_set:
                continue
            valid = True
            for w in current_words:
                if (candidate ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                current_words.append(candidate)
                word_set.add(candidate)
                
        return current_words
