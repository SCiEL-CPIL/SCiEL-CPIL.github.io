import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Pre-compute popcount table for 17-bit integers
    popcount = np.zeros(1 << 17, dtype=np.uint8)
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

    # Build generator polynomial g(x) of degree 8
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

    # Shorten: keep words with parity bit = 0
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

    # Step 2: Targeted nonlinear extension using truly vectorized NumPy
    # Generate all 17-bit words at Hamming distance exactly 6 from zero
    # C(17,6) = 12376 candidates
    
    bit_positions = list(range(17))
    candidate_pool = []
    for combo in combinations(bit_positions, 6):
        val = 0
        for pos in combo:
            val |= (1 << pos)
        candidate_pool.append(val)
    
    candidate_pool = np.array(candidate_pool, dtype=np.uint32)
    
    # Also generate candidates at distance 7, 8, 9 for more coverage
    for r in [7, 8, 9]:
        for combo in combinations(bit_positions, r):
            val = 0
            for pos in combo:
                val |= (1 << pos)
            candidate_pool = np.append(candidate_pool, val)
    
    candidate_pool = np.array(candidate_pool, dtype=np.uint32)
    
    code_set = set(current_code)
    code_list = list(current_code)
    
    def vectorized_check(candidates, code_arr):
        """
        Check which candidates are compatible with the current code.
        Returns list of valid candidates using vectorized popcount.
        """
        if len(candidates) == 0:
            return []
        
        valid_candidates = []
        
        # Process candidates in batches
        batch_size = 500
        for start in range(0, len(candidates), batch_size):
            if time.time() > deadline - 5.0:
                break
            end = min(start + batch_size, len(candidates))
            batch = candidates[start:end]
            
            # Skip candidates already in the code
            mask = np.array([c not in code_set for c in batch], dtype=bool)
            if not np.any(mask):
                continue
            batch = batch[mask]
            if len(batch) == 0:
                continue
            
            # Compute XOR: batch (m,1) XOR code (1,n) -> (m,n)
            xor_matrix = batch[:, np.newaxis] ^ code_arr[np.newaxis, :]
            
            # Vectorized popcount: split into bytes and look up
            # xor_matrix is uint32, we need popcount of each element
            # Use the precomputed table via fancy indexing
            # Reshape to handle in chunks to limit memory
            chunk_size = 200
            for cs in range(0, len(batch), chunk_size):
                if time.time() > deadline - 5.0:
                    break
                ce = min(cs + chunk_size, len(batch))
                chunk = xor_matrix[cs:ce]  # shape (chunk_len, code_len)
                
                # Get indices for popcount lookup
                flat = chunk.flatten()
                pcs = popcount[flat]  # shape (chunk_len * code_len,)
                pcs = pcs.reshape(chunk.shape)
                
                # For each row, check if min popcount >= 6
                row_mins = pcs.min(axis=1)
                valid_rows = np.where(row_mins >= 6)[0]
                
                for idx in valid_rows:
                    valid_candidates.append(batch[cs + idx])
        
        return valid_candidates

    # Iteratively add valid candidates
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > deadline - 5.0:
            break
        
        code_arr = np.array(code_list, dtype=np.uint32)
        
        valid_new = vectorized_check(candidate_pool, code_arr)
        
        if not valid_new:
            break
        
        for w in valid_new:
            code_set.add(w)
            code_list.append(w)
        
        added = len(valid_new)
        
        if len(code_list) > best_size:
            best_size = len(code_list)
            best_code = list(code_list)
        
        # Remove added candidates from pool
        if added > 0:
            added_arr = np.array(valid_new, dtype=np.uint32)
            mask = np.isin(candidate_pool, added_arr)
            candidate_pool = candidate_pool[~mask]
        
        if len(candidate_pool) == 0:
            break
    
    # Step 3: Local search with random candidates and swaps
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 1.0:
        code_set = set(best_code)
        code_list = list(best_code)
        best_size = len(code_list)
        best_code_local = list(code_list)
        
        max_iters = 50000
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            w = random.randint(0, 0x1FFFF)
            
            if w in code_set:
                continue
            
            ok = True
            for c in code_list:
                if popcount_fast(w ^ c) < 6:
                    ok = False
                    break
            
            if ok:
                code_set.add(w)
                code_list.append(w)
                if len(code_list) > best_size:
                    best_size = len(code_list)
                    best_code_local = list(code_list)
            else:
                # Try swap: remove a random word, see if w fits
                if len(code_list) > 50 and random.random() < 0.3:
                    idx = random.randint(0, len(code_list) - 1)
                    u = code_list[idx]
                    code_set.remove(u)
                    code_list.pop(idx)
                    
                    ok2 = True
                    for c in code_list:
                        if popcount_fast(w ^ c) < 6:
                            ok2 = False
                            break
                    
                    if ok2 and w not in code_set:
                        code_set.add(w)
                        code_list.append(w)
                        if len(code_list) > best_size:
                            best_size = len(code_list)
                            best_code_local = list(code_list)
                    else:
                        code_set.add(u)
                        code_list.insert(idx, u)
        
        if best_code_local:
            best_code = best_code_local
            best_size = len(best_code)
    
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
