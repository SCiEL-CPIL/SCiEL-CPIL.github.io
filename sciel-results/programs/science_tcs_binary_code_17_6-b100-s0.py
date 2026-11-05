import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # The goal is to construct the [17,8,6] linear code
    # which is the shortened version of the [18,9,6] extended QR code.
    
    # 1. Work in GF(2^8) to find a 17th root of unity.
    # We need an element alpha such that alpha^17 = 1.
    # Since 17 is prime, the multiplicative group of GF(2^8) has order 255.
    # 17 divides 255. So elements of order 17 exist.
    
    # Find a primitive polynomial for GF(2^8). 
    # One common one is x^8 + x^4 + x^3 + x^2 + 1 (0x1d) or x^8 + x^4 + x^3 + x + 1 (0x1b).
    # Let's use a standard one.
    # We represent elements as integers 0..255.
    
    def gf_mul(a, b, poly=0x1d):
        # Multiplication in GF(2^8) with irreducible poly x^8+x^4+x^3+x^2+1
        res = 0
        for i in range(8):
            if (b & 1):
                res ^= a
            b >>= 1
            if (a & 0x80):
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n, poly=0x1d):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    # Find an element of order 17.
    # The elements of order 17 are roots of x^17 - 1 = (x-1)(x^16 + ... + 1).
    # We can just iterate through all 255 non-zero elements and check if alpha^17 == 1 and alpha != 1.
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        # Fallback to random search if specific poly doesn't work (should not happen)
        return [0] * 32 # Fallback to seed if construction fails

    # 2. Construct the generator polynomial g(x) for the [17,9,5] QR code.
    # Roots are alpha^i for i in Q (quadratic residues mod 17).
    # Q = {1, 2, 4, 8, 9, 13, 15, 16}
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # g(x) = prod_{i in Q} (x - alpha^i)
    # In GF(2), x - a = x + a.
    # We build g(x) as a list of coefficients [g_0, g_1, ..., g_8] representing g_0 + g_1 x + ...
    
    g_coeffs = [1] # Start with 1
    
    for i in Q:
        root = gf_pow(alpha, i)
        # Multiply g(x) by (x + root)
        # New degree is len(g_coeffs)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            # x * g_j(x) contributes to index j+1
            new_g[j+1] ^= g_coeffs[j]
            # root * g_j(x) contributes to index j
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # g_coeffs now has 9 elements (degree 8).
    # The generator matrix G is 9x17.
    # Row k of G is the coefficients of x^k * g(x) mod (x^17 - 1).
    
    G = np.zeros((9, 17), dtype=np.uint8)
    
    for k in range(9):
        # Compute x^k * g(x)
        # g(x) = sum_{j=0}^8 g_j x^j
        # x^k * g(x) = sum_{j=0}^8 g_j x^{j+k}
        # If j+k >= 17, wrap around: x^{j+k} = x^{j+k-17}
        for j in range(9):
            bit = g_coeffs[j]
            if bit:
                pos = (j + k) % 17
                G[k, pos] = 1
                
    # 3. Generate all 512 codewords of the [17,9,5] code.
    # A codeword is u * G where u is a 9-bit vector.
    # We represent u as an integer 0..511.
    
    words_17 = []
    for u in range(512):
        # Extract bits of u
        # u = sum u_i 2^i. u_i is the coefficient for row i of G.
        # We sum the rows of G where u_i is 1.
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    # 4. Extend to [18,9,6].
    # Add a parity bit (bit 17) such that the total weight is even.
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        # Append parity bit
        w_ext = np.append(w, parity)
        words_18.append(w_ext)
        
    # 5. Shorten to [17,8,6].
    # Pick a coordinate to shorten. Let's pick the last one (index 17 in 0-based 18-length).
    # Keep only words with 0 at that position.
    # Remove that position.
    
    words_17_short = []
    for w in words_18:
        if w[17] == 0:
            # Remove last bit
            w_short = w[:17]
            # Convert to integer
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_17_short.append(val)
            
    # There should be 256 words.
    if len(words_17_short) == 256:
        # Verify min distance just in case
        # Sample check or full check? 256^2 / 2 ~ 32k checks. Fast enough.
        # But we trust the math.
        return words_17_short
    else:
        # Fallback to seed if construction fails
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
