import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 17
    MIN_DIST = 6
    
    # GF(256) construction for the QR code.
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    
    def gf_mul(a, b, poly=0x11D):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n, poly=0x11D):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base, poly)
            base = gf_mul(base, base, poly)
            n >>= 1
        return res

    def get_primitive_17th_root():
        # 255 = 15 * 17
        # Primitive element alpha. Let's test 2.
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    current_code = None
    
    if beta:
        # Build the generator polynomial
        poly_coeffs = [1]
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i+1] = poly_coeffs[i]
            for i in range(len(poly_coeffs)):
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)
            
        g_bits = []
        for c in poly_coeffs:
            if c == 1:
                g_bits.append(1)
            elif c == 0:
                g_bits.append(0)
            else:
                g_bits.append(1)
        
        if len(g_bits) == 9 and g_bits[0] == 1:
            # Generate [17, 9] code
            codewords_17 = []
            for u in range(512):
                c = 0
                for k in range(9):
                    if (u >> k) & 1:
                        for j in range(9):
                            if g_bits[j]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)
            
            # Extend to 18
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)
            
            # Shorten (fix bit 17 to 0)
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & ((1 << 17) - 1))
            
            # Remove duplicates
            shortened_codewords = list(set(shortened_codewords))
            
            # Verify
            if len(shortened_codewords) >= 256:
                min_d = 100
                for i in range(len(shortened_codewords)):
                    for j in range(i+1, len(shortened_codewords)):
                        d = (shortened_codewords[i] ^ shortened_codewords[j]).bit_count()
                        if d < min_d:
                            min_d = d
                if min_d >= 6:
                    current_code = set(shortened_codewords)
    
    if current_code is None:
        # Fallback: Greedy from scratch
        current_code = set()
        current_code.add(0)
        
        random.seed(42)
        candidates = list(range(1, 1 << 17))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline - 10.0:
                break
            valid = True
            for w in current_code:
                if (cand ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                current_code.add(cand)
    
    code_set = set(current_code)
    
    # Systematic search for additional codewords (holes)
    # We look for words of weight 4, 5, and 6 that are at distance >= 6 from all existing codewords.
    
    # Precompute existing codewords for fast access
    existing_list = list(code_set)
    
    # Generate all candidates of weight 4, 5, 6
    candidates = []
    # Weight 4
    for comb in combinations(range(17), 4):
        w = 0
        for i in comb:
            w |= (1 << i)
        candidates.append(w)
    # Weight 5
    for comb in combinations(range(17), 5):
        w = 0
        for i in comb:
            w |= (1 << i)
        candidates.append(w)
    # Weight 6
    for comb in combinations(range(17), 6):
        w = 0
        for i in comb:
            w |= (1 << i)
        candidates.append(w)
    
    # Filter candidates that are valid against the initial code
    valid_candidates = []
    for cand in candidates:
        if cand in code_set:
            continue
        valid = True
        for w in existing_list:
            if (cand ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            valid_candidates.append(cand)
    
    # Now, try to add these valid candidates to the code set.
    # We must ensure they are also valid against each other.
    # Since the number of valid candidates is likely small (hundreds or less),
    # we can do a greedy pass or a small search.
    
    # Sort by weight? Or just iterate.
    # To maximize count, we want to pick as many as possible.
    # This is a Maximum Independent Set problem on the conflict graph of valid_candidates.
    # But the graph is likely sparse.
    
    # Greedy approach:
    # Iterate through valid_candidates, add if compatible with all currently in code_set.
    # To improve, we can try multiple orderings or restarts.
    
    best_size = len(code_set)
    best_set = set(code_set)
    
    # Try a few random orderings of valid_candidates
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
            
        current_set = set(best_set)
        shuffled_cands = valid_candidates[:]
        random.shuffle(shuffled_cands)
        
        for cand in shuffled_cands:
            if cand in current_set:
                continue
            valid = True
            for w in current_set:
                if (cand ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                current_set.add(cand)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Final check and return
    return list(best_set)
