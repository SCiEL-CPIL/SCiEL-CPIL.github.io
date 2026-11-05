import numpy as np
import time
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    TIME_LIMIT = 55
    
    # [17,8,6] code generator matrix (shortened extended QR code)
    G = [
        0b100000000111000110,
        0b010000000110101100,
        0b001000000101101010,
        0b000100000011110110,
        0b000010000100110001,
        0b000001000010111111,
        0b000000100101011001,
        0b000000010010100111,
    ]
    
    codewords = []
    for mask in range(256):
        word = 0
        for i in range(8):
            if mask & (1 << i):
                word ^= G[i]
        codewords.append(word)
    
    # Verify minimum distance
    min_d = min((codewords[i] ^ codewords[j]).bit_count() 
                for i in range(256) for j in range(i+1, 256))
    
    if min_d < MIN_DIST:
        # Fallback: greedy construction
        codewords = [0]
        for c in range(1, 1 << N):
            if all((c ^ w).bit_count() >= MIN_DIST for w in codewords):
                codewords.append(c)
            if len(codewords) >= 256:
                break
    
    code_set = set(codewords)
    
    # Generate candidates at distance 5 from each codeword
    candidates = []
    seen = set()
    all_bits = list(range(N))
    
    for w in codewords:
        for comb in combinations(all_bits, 5):
            cand = w
            for b in comb:
                cand ^= (1 << b)
            if cand not in code_set and cand not in seen:
                seen.add(cand)
                candidates.append(cand)
    
    # Sort by weight (ascending) - lighter words first
    candidates.sort(key=lambda x: x.bit_count())
    
    # Prepare for vectorized distance checks
    bit_pos = np.arange(N)
    code_arr = np.array([[ (w >> i) & 1 for i in range(N)] for w in codewords], dtype=np.uint8)
    
    # Pre-allocate
    max_size = 300
    code_ext = np.zeros((max_size, N), dtype=np.uint8)
    code_ext[:len(codewords)] = code_arr
    code_size = len(codewords)
    
    for cand in candidates:
        if time.time() - start_time > TIME_LIMIT:
            break
        if code_size >= max_size:
            break
        
        cand_arr = ((cand >> bit_pos) & 1).astype(np.uint8)
        
        xor = code_ext[:code_size] ^ cand_arr
        dists = xor.sum(axis=1)
        
        if dists.min() >= MIN_DIST:
            code_ext[code_size] = cand_arr
            code_size += 1
            code_set.add(cand)
    
    # Convert back to list of integers
    result = []
    for i in range(code_size):
        w = 0
        for bit in range(N):
            if code_ext[i, bit]:
                w |= (1 << bit)
        result.append(w)
    
    return result
