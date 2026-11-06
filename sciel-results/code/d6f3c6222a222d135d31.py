import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    # GF(2^8) arithmetic
    PP = 0x11d
    
    def gf_mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            if a & 0x100:
                a ^= PP
            b >>= 1
        return result

    def gf_pow(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mul(result, a)
            a = gf_mul(a, a)
            n >>= 1
        return result

    # Find alpha of order 17
    alpha = gf_pow(2, 15)
    if gf_pow(alpha, 17) != 1:
        for e in range(1, 256):
            if gf_pow(e, 17) == 1 and e != 1:
                alpha = e
                break

    NQR = [3, 5, 6, 7, 10, 11, 12, 14]
    
    g_poly = [1]
    for r in NQR:
        root = gf_pow(alpha, r)
        new_g = [0] * (len(g_poly) + 1)
        for i in range(len(g_poly)):
            new_g[i] ^= g_poly[i] * root
            new_g[i+1] ^= g_poly[i]
        g_poly = new_g
        
    g_coeffs = [1 if c else 0 for c in g_poly]
    
    codewords = []
    for msg in range(512):
        m_poly = [0] * 9
        for i in range(9):
            if msg & (1 << i):
                m_poly[i] = 1
        
        c_poly = [0] * 17
        for i in range(9):
            if m_poly[i]:
                for j in range(9):
                    if g_coeffs[j]:
                        c_poly[i+j] ^= 1
        
        word = 0
        for k in range(17):
            if c_poly[k]:
                word |= (1 << k)
        codewords.append(word)
        
    qr_code = [w for w in codewords if bin(w).count('1') % 2 == 0]
    current_selected = qr_code[:]
    
    # Find valid candidates
    sel_set = set(current_selected)
    valid_candidates = []
    
    # Optimization: Use numpy for batch processing if possible, 
    # but Python loop with early break is often faster for small N.
    
    for w in range(1 << N):
        if w in sel_set:
            continue
        
        is_valid = True
        for s in current_selected:
            if (w ^ s).bit_count() < MIN_DIST:
                is_valid = False
                break
        
        if is_valid:
            valid_candidates.append(w)
            
    # Now try to add from valid_candidates
    # We want to add as many as possible.
    # Greedy:
    added = []
    for w in valid_candidates:
        if time.time() > deadline - 5:
            break
        is_valid = True
        for s in added:
            if (w ^ s).bit_count() < MIN_DIST:
                is_valid = False
                break
        if is_valid:
            added.append(w)
            
    final_set = current_selected + added
    
    return final_set
