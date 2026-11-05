import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Step 1: Construct the 256-word linear code
    poly = 0x11D  # Primitive polynomial for GF(2^8)
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
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
    for a in range(2, 256):
        if gf_pow(a, 255) == 1 and gf_pow(a, 85) != 1:
            alpha = a
            break
            
    if alpha is None:
        return [0]
        
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1
                
    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)
        
    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)
            
    if not words_short:
        return [0]
        
    current_code = np.array(words_short, dtype=np.int64)
    
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)
    
    best_code = current_code.copy()
    best_size = len(current_code)
    
    all_words = np.arange(2**17, dtype=np.int64)
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        rng = np.random.default_rng(seed=restart * 7919 + 12345)
        
        current_list = list(current_code)
        current_array = np.array(current_list, dtype=np.int64)
        current_set = set(current_list)
        
        # Candidate ordering
        if restart % 5 == 0:
            candidates = rng.permutation(all_words)
        elif restart % 5 == 1:
            wts = popcount_table.copy()
            candidates = np.argsort(wts)
        elif restart % 5 == 2:
            wts = popcount_table.copy()
            candidates = np.argsort(-wts)
        elif restart % 5 == 3:
            wts = popcount_table.copy()
            candidates = np.argsort(np.abs(wts - 8.5))
        else:
            # Random with slight bias toward medium weights
            wts = popcount_table.astype(float)
            scores = np.exp(-((wts - 8.5) ** 2) / 20.0)
            candidates = rng.choice(all_words, size=len(all_words), replace=False, p=scores/scores.sum())
        
        for idx in range(len(candidates)):
            if time.time() > deadline - 0.5:
                break
            
            w = int(candidates[idx])
            
            if w in current_set:
                continue
            
            # Check min distance to all current codewords
            diffs = current_array ^ w
            dists = popcount_table[diffs]
            
            if np.any(dists < 6):
                continue
            
            current_list.append(w)
            current_set.add(w)
            current_array = np.append(current_array, w)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_code = np.array(current_list, dtype=np.int64)
            
    return list(best_code)
