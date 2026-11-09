import numpy as np
import time

def construct():
    deadline = time.time() + 50
    
    # GF(2^8) with primitive polynomial 0x11D
    gf_mul = np.zeros((256, 256), dtype=np.int32)
    for a in range(256):
        for b in range(256):
            prod = 0
            x = a
            y = b
            for _ in range(8):
                if y & 1:
                    prod ^= x
                y >>= 1
                x <<= 1
                if x & 0x100:
                    x ^= 0x11D
            gf_mul[a, b] = prod
    
    def gf_pow(base, exp):
        if exp == 0:
            return 1
        result = 1
        for _ in range(exp):
            result = gf_mul[result, base]
        return result
    
    # alpha = 2^15, a primitive 17th root of unity
    alpha = gf_pow(2, 15)
    assert gf_pow(alpha, 17) == 1
    
    # Roots: alpha^i for i in {1,2,4,8,16,15,13,9}
    root_indices = [1, 2, 4, 8, 16, 15, 13, 9]
    roots = [gf_pow(alpha, i) for i in root_indices]
    
    # Build g(x) = prod (x + r_i) in GF(2^8)[x], should have GF(2) coefficients
    g = [1]  # constant polynomial 1
    for r in roots:
        new_g = [0] * (len(g) + 1)
        for i, c in enumerate(g):
            new_g[i] ^= gf_mul(c, r)
            new_g[i+1] ^= c
        g = new_g
    
    # Verify all coefficients are 0 or 1
    for c in g:
        assert c in (0, 1), f"Non-binary coefficient: {c}"
    
    # g is [g0, g1, ..., g8] where g(x) = g0 + g1*x + ... + g8*x^8
    # Generator matrix for cyclic [17,9,5] code
    gen = np.zeros((9, 17), dtype=np.int32)
    for i in range(9):
        for j in range(9):
            pos = i + j
            if pos < 17:
                gen[i, pos] = g[j]
            else:
                gen[i, pos - 17] ^= g[j]
    
    # Extend to [18,9,6]: add parity bit
    ext_gen = np.zeros((9, 18), dtype=np.int32)
    ext_gen[:, :17] = gen
    for i in range(9):
        ext_gen[i, 17] = int(np.sum(gen[i]) % 2)
    
    # Generate all 512 codewords
    codewords_18 = []
    for msg in range(512):
        cw = np.zeros(18, dtype=np.int32)
        for i in range(9):
            if (msg >> i) & 1:
                cw ^= ext_gen[i]
        val = 0
        for j in range(18):
            val |= int(cw[j]) << j
        codewords_18.append(val)
    
    # Shorten: keep codewords with bit 17 = 0, remove bit 17
    codewords_17 = []
    for cw in codewords_18:
        if (cw >> 17) & 1 == 0:
            codewords_17.append(cw & 0x1FFFF)
    
    return codewords_17
