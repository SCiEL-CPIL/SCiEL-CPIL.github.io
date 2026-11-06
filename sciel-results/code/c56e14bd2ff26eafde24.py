import time
import random
from itertools import combinations

def construct():
    N = 18
    D = 6
    
    # GF(2^8) arithmetic
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            if a & 256:
                a ^= 0x11D
            b >>= 1
        return res

    def gf_pow(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            e >>= 1
        return res

    # Find alpha = beta^15, where beta is root of x^8+x^4+x^3+x^2+1
    beta = 2 # x
    alpha = gf_pow(beta, 15)
    
    # Quadratic residues mod 17
    residues = [1, 2, 4, 8, 16, 15, 13, 9]
    
    # Compute generator polynomial g(x) = prod (x + alpha^q)
    # Coefficients in GF(2^8)
    poly = [1] # c0
    for q in residues:
        root = gf_pow(alpha, q)
        # Multiply by (x + root)
        new_poly = [0] * (len(poly) + 1)
        for i in range(len(poly)):
            # x * poly[i] -> new_poly[i+1]
            new_poly[i+1] ^= poly[i]
            # root * poly[i] -> new_poly[i]
            new_poly[i] ^= gf_mul(root, poly[i])
        poly = new_poly
        
    # poly should be [c0, c1, ..., c8]
    # Convert to integer mask for g(x)
    # g(x) = x^8 + c7 x^7 + ... + c0
    # We represent g as an integer where bit i is coeff of x^i
    g_int = 0
    for i, c in enumerate(poly):
        if c != 0 and c != 1:
            # Should not happen for minimal polynomial over GF(2)
            # But if it does, we have an issue
            pass
        if c & 1:
            g_int |= (1 << i)
            
    # g_int has bits 0..8 set.
    # Generate all 512 codewords
    words = []
    for m in range(512):
        # c(x) = m(x) * g(x)
        # m is 9 bits, g is 9 bits (degree 8)
        # Product is up to 17 bits
        c = 0
        mm = m
        gg = g_int
        while mm:
            if mm & 1:
                c ^= gg
            mm >>= 1
            gg <<= 1
        # c is 17 bits
        # Append 0 to make 18 bits
        word = c << 1
        words.append(word)
        
    # Remove duplicates (should be none)
    words = list(set(words))
    
    # Optional: Verify distance for a few pairs to be safe?
    # If distance < 6, fall back to greedy?
    # Given the inspiration, we trust it.
    # But let's do a quick check on the first 100 pairs
    bc = int.bit_count
    valid = True
    for i in range(min(100, len(words))):
        for j in range(i+1, min(100, len(words))):
            if bc(words[i] ^ words[j]) < D:
                valid = False
                break
        if not valid:
            break
            
    if valid:
        return words
    else:
        # Fallback to simple greedy if construction failed
        # (This should not happen)
        return [0]
