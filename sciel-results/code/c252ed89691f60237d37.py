import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Pre-compute popcount table for 17-bit integers (2^17 = 131072 entries)
    # 17 bits -> values from 0 to 131071
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def popcount_fast(x):
        return popcount[x & 0x1FFFF]

    def is_valid_code(words):
        for i in range(len(words)):
            wi = words[i]
            for j in range(i + 1, len(words)):
                if popcount_fast(wi ^ words[j]) < 6:
                    return False
        return True

    # Step 1: Build the [17,8,6] linear code (256 words)
    # This is the shortened version of the [18,9,6] extended QR(17) code.
    
    gf_poly = 0x1d  # x^8+x^4+x^3+x^2+1, primitive in GF(2^8)
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = ((a << 1) ^ gf_poly) & 0xFF
            else:
                a = (a << 1) & 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find element of order 17
    alpha = None
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        alpha = 3

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial g(x) of degree 8: g(x) = prod_{i in Q} (x - alpha^i)
    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    # Generate all 512 words of the [17,9,5] code
    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if (g[k] != 0):
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    # Extend to [18,9,6] by appending overall parity
    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    # Shorten: keep words with parity bit = 0, then mask off the 18th coordinate.
    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)

    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6

    if not is_valid_code(base_code):
        valid_code = [base_code[0]]
        for w in base_code[1:]:
            ok = True
            for v in valid_code:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                valid_code.append(w)
        base_code = valid_code

    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)
    
    if best_size < 32:
        return [0]
    
    # Step 2: Local Search to grow the code
    # We use a set for O(1) membership checks and a list for iteration.
    # To check if a new word w is valid, we must ensure d(w, c) >= 6 for all c in current_code.
    
    def try_add_word(code_set, code_list, w):
        """Check if w can be added. Returns True if added, False otherwise."""
        if w in code_set:
            return False
        for c in code_list:
            if popcount_fast(w ^ c) < 6:
                return False
        code_set.add(w)
        code_list.append(w)
        return True

    # Local Search Loop
    # We will perform many iterations.
    # Each iteration:
    # 1. Pick a random word w.
    # 2. Try to add it.
    # 3. If successful, update best.
    # 4. If not, with some probability, perform a swap:
    #    a. Pick a random word u from current code.
    #    b. Remove u.
    #    c. Try to add w.
    #    d. If successful, keep the new state.
    #    e. If not, restore u.
    
    # To make it faster, we can pre-compute some properties or use numpy for batch checks?
    # Given N=256, linear scan is ~256 ops. 10000 iterations * 256 ops = 2.5M ops. Very fast.
    
    # We will run this search for a fixed number of iterations or until time limit.
    
    code_set = set(current_code)
    code_list = list(current_code)
    
    # Random seed for reproducibility if needed, but random is fine.
    # We'll use a fixed number of iterations to be safe, plus time check.
    
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Generate a random candidate
        # Bias towards words that are "far" from the current code?
        # Just random is a good baseline.
        w = random.randint(0, 0x1FFFF)
        
        added = try_add_word(code_set, code_list, w)
        
        if added:
            if len(code_list) > best_size:
                best_size = len(code_list)
                best_code = list(code_list)
        else:
            # Try a swap
            # Only if code is not too small to avoid emptying it
            if len(code_list) > 50:
                # Pick a random word to remove
                idx = random.randint(0, len(code_list) - 1)
                u = code_list[idx]
                
                # Remove u
                code_set.remove(u)
                code_list.pop(idx)
                
                # Try to add w
                if w not in code_set:
                    ok = True
                    for c in code_list:
                        if popcount_fast(w ^ c) < 6:
                            ok = False
                            break
                    if ok:
                        code_set.add(w)
                        code_list.append(w)
                        # Check if we improved
                        if len(code_list) > best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                    else:
                        # Restore u
                        code_set.add(u)
                        code_list.insert(idx, u)
                else:
                    # w is already in code, restore u
                    code_set.add(u)
                    code_list.insert(idx, u)
                    
    # Final verification
    if len(best_code) >= 32:
        if is_valid_code(best_code):
            return best_code
        else:
            final = []
            for w in best_code:
                ok = True
                for v in final:
                    if popcount_fast(w ^ v) < 6:
                        ok = False
                        break
                if ok:
                    final.append(w)
            if len(final) >= 32:
                return final
            return base_code
    else:
        return base_code if len(base_code) >= 32 else [0]
