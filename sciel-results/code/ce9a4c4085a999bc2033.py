import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N

    # GF(256) arithmetic with polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    current_code = set()

    if beta:
        # Build minimal polynomial of beta over GF(2)
        poly_coeffs = [1]
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i + 1] = poly_coeffs[i]
            for i in range(len(poly_coeffs)):
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)

        # Check that coefficients are 0 or 1
        g_bits = []
        valid_poly = True
        for c in poly_coeffs:
            if c == 0:
                g_bits.append(0)
            elif c == 1:
                g_bits.append(1)
            else:
                valid_poly = False
                break

        if valid_poly and len(g_bits) == 9 and g_bits[0] == 1:
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

            # Verify min distance
            min_d = 100
            for i in range(len(shortened_codewords)):
                for j in range(i + 1, len(shortened_codewords)):
                    d = (shortened_codewords[i] ^ shortened_codewords[j]).bit_count()
                    if d < min_d:
                        min_d = d

            if min_d >= 6 and len(shortened_codewords) >= 256:
                current_code = set(shortened_codewords)

    # If we didn't get a good code from QR, fall back to greedy
    if len(current_code) < 10:
        code = set()
        code.add(0)
        random.seed(42)
        candidates = list(range(1, TOTAL))
        random.shuffle(candidates)
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            valid = True
            for w in code:
                if (cand ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                code.add(cand)
        current_code = code

    # Now try to expand the code using randomized local search
    random.seed(123)
    code_list = list(current_code)
    
    # Phase 1: Try to add random words
    added_count = 0
    max_add_attempts = 200000
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < max_add_attempts:
        cand = random.randint(0, TOTAL - 1)
        if cand in current_code:
            attempts += 1
            continue
        
        valid = True
        for w in code_list:
            if (cand ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            current_code.add(cand)
            code_list.append(cand)
            added_count += 1
        attempts += 1

    # Phase 2: Try swap-based local search to grow the code
    # For each codeword, try removing it and adding a new one that fits better
    swap_improved = 0
    max_swap_rounds = 50000
    swap_attempts = 0
    while time.time() < deadline - 5.0 and swap_attempts < max_swap_rounds:
        if not code_list:
            break
        
        # Pick a random codeword to potentially remove
        idx = random.randint(0, len(code_list) - 1)
        old_word = code_list[idx]
        
        # Temporarily remove it
        current_code.discard(old_word)
        code_list.pop(idx)
        
        # Try to find a new word that fits with the remaining code
        new_word = None
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            cand = random.randint(0, TOTAL - 1)
            if cand in current_code:
                continue
            valid = True
            for w in code_list:
                if (cand ^ w).bit_count() < 6:
                    valid = False
                    break
            if valid:
                new_word = cand
                break
        
        if new_word is not None:
            # Check if the new word is at least as far from old_word's neighbors
            # Simple heuristic: just accept if it fits
            current_code.add(new_word)
            code_list.append(new_word)
            swap_improved += 1
        else:
            # Put the old word back
            current_code.add(old_word)
            code_list.append(old_word)
        
        swap_attempts += 1

    # Phase 3: Another round of random additions
    attempts = 0
    while time.time() < deadline - 2.0 and attempts < 100000:
        cand = random.randint(0, TOTAL - 1)
        if cand in current_code:
            attempts += 1
            continue
        
        valid = True
        for w in code_list:
            if (cand ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            current_code.add(cand)
            code_list.append(cand)
        attempts += 1

    return list(current_code)
