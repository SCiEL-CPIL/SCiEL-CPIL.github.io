import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Pre-compute popcount table for 17-bit integers
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)
    popcount_np = np.array(popcount, dtype=np.int8)

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
    # GF(2^8) construction for QR(17)
    gf_poly = 0x1d
    
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

    alpha = None
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break
    if alpha is None:
        alpha = 3

    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if g[k] != 0:
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)

    base_code = list(set(words_17_8_6))
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

    # Step 2: Generate Candidates
    # Focus on weights 6 and 7, as they are the most likely to fit.
    # Weight 6: C(17,6) = 12376
    # Weight 7: C(17,7) = 19448
    # Total ~32k candidates.
    
    candidates = []
    # Weight 6
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m in range(l+1, 17):
                        for n in range(m+1, 17):
                            w = (1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m) | (1 << n)
                            candidates.append(w)
    # Weight 7
    for i in range(17):
        for j in range(i+1, 17):
            for k in range(j+1, 17):
                for l in range(k+1, 17):
                    for m in range(l+1, 17):
                        for n in range(m+1, 17):
                            for o in range(n+1, 17):
                                w = (1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m) | (1 << n) | (1 << o)
                                candidates.append(w)
    
    # Remove any candidates already in the base code
    base_set = set(current_code)
    candidates = [c for c in candidates if c not in base_set]
    
    cands_np = np.array(candidates, dtype=np.int64)
    code_np = np.array(current_code, dtype=np.int64)
    
    # Shuffle candidates for randomization
    np.random.shuffle(cands_np)
    
    # Helper to check if a single candidate fits
    def fits(w, code_arr):
        # code_arr is numpy array
        return np.all(popcount_np[w ^ code_arr] >= 6)

    # Phase 1: Greedy Add
    # Use a list of indices for active candidates
    active_indices = list(np.arange(len(cands_np)))
    np.random.shuffle(active_indices)
    
    while time.time() < deadline - 15.0 and len(active_indices) > 0:
        # Check first active candidate
        idx = active_indices[0]
        w_new = int(cands_np[idx])
        
        # Check if w_new fits with current_code
        if fits(w_new, code_np):
            current_code.append(w_new)
            code_np = np.append(code_np, w_new)
            active_indices.pop(0)
            
            if len(current_code) > best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
            # Filter active_indices to remove those that don't fit with w_new
            # This is expensive if done every time. 
            # Given the small size, we can skip this and just let the next iteration fail.
        else:
            active_indices.pop(0)
            
    # Phase 2: Local Search (Remove 1, Add 2)
    # While time remains, try to improve the code.
    
    # Re-initialize active list based on current_code
    code_set = set(current_code)
    active_indices = [i for i in range(len(cands_np)) if int(cands_np[i]) not in code_set]
    
    # Shuffle active indices for randomness in selection
    random.shuffle(active_indices)
    
    while time.time() < deadline - 2.0:
        if len(current_code) <= 32:
            break
            
        # Pick a random word to remove
        idx_rm = random.randint(0, len(current_code) - 1)
        w_old = current_code[idx_rm]
        
        # Temporarily remove
        current_code.pop(idx_rm)
        code_np = np.array(current_code, dtype=np.int64)
        
        # Find candidates that fit with the reduced code
        if len(active_indices) == 0:
            current_code.append(w_old)
            code_np = np.append(code_np, w_old)
            break
            
        active_cands = cands_np[active_indices]
        dists = popcount_np[active_cands[:, None] ^ code_np[None, :]]
        min_dists = np.min(dists, axis=1)
        valid_rel_indices = np.where(min_dists >= 6)[0]
        
        if len(valid_rel_indices) >= 2:
            # We can add 2 words!
            # Pick the first two
            idx1 = valid_rel_indices[0]
            idx2 = valid_rel_indices[1]
            
            w_new1 = int(active_cands[idx1])
            w_new2 = int(active_cands[idx2])
            
            current_code.append(w_new1)
            current_code.append(w_new2)
            
            # Update active_indices
            glob_idx1 = active_indices[idx1]
            glob_idx2 = active_indices[idx2]
            
            # Remove from active_indices list
            active_indices.remove(glob_idx1)
            active_indices.remove(glob_idx2)
            
            code_np = np.array(current_code, dtype=np.int64)
            
            if len(current_code) > best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        elif len(valid_rel_indices) == 1:
            # Add 1 word (Swap)
            idx1 = valid_rel_indices[0]
            w_new1 = int(active_cands[idx1])
            
            current_code.append(w_new1)
            
            glob_idx1 = active_indices[idx1]
            active_indices.remove(glob_idx1)
            
            code_np = np.array(current_code, dtype=np.int64)
            
        else:
            # Restore w_old
            current_code.append(w_old)
            code_np = np.append(code_np, w_old)
            
    # Final Verification
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
