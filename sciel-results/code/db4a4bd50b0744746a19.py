import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # 1. Construct [17, 8, 6] linear code
    # GF(256) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    
    def gf_mul(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
        return p

    def gf_pow(a, n):
        result = 1
        for _ in range(n):
            result = gf_mul(result, a)
        return result

    def poly_mul(p, q):
        res = [0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                res[i+j] ^= gf_mul(pi, qj)
        return res

    # QR(17)
    qr = [1, 2, 4, 8, 9, 13, 15, 16]
    g = [1]
    alpha = 2
    for i in qr:
        root = gf_pow(alpha, i)
        g = poly_mul(g, [root, 1])
        
    # g is degree 8, coeffs in GF(2) (0 or 1)
    # Verify
    # for coeff in g:
    #     assert coeff in [0, 1]
    
    # Generate [17, 9, 5] code
    code_words = []
    for m in range(512):
        m_coeffs = []
        temp = m
        for _ in range(9):
            m_coeffs.append(temp & 1)
            temp >>= 1
        c_coeffs = [0] * 17
        for i in range(9):
            if m_coeffs[i]:
                for j in range(9):
                    c_coeffs[i+j] ^= g[j]
        word = 0
        for i in range(17):
            if c_coeffs[i] != 0:
                word |= (1 << i)
        code_words.append(word)
        
    # Filter even weight
    even_code = [w for w in code_words if bin(w).count('1') % 2 == 0]
    
    # 2. Find candidates
    # Convert to numpy
    code_bits = np.array([[ (w >> i) & 1 for i in range(17) ] for w in even_code], dtype=np.uint8)
    even_set = set(even_code)
    
    candidates = []
    
    # Chunked processing
    chunk_size = 1000
    total_words = 1 << 17
    
    for start in range(0, total_words, chunk_size):
        end = min(start + chunk_size, total_words)
        # Generate bits for this chunk
        # Optimize: pre-generate all bits?
        # Let's just generate on the fly for the chunk
        chunk_words = list(range(start, end))
        chunk_bits = np.array([[(w >> i) & 1 for i in range(17)] for w in chunk_words], dtype=np.uint8)
        
        # Exclude words already in even_code
        # We can mask them out later or just check min dist.
        # If w is in even_code, dist to itself is 0, so min dist < 6.
        # So they will be naturally excluded.
        
        # Compute distances
        # chunk_bits: (N, 17)
        # code_bits: (256, 17)
        # diffs: (N, 256, 17)
        diffs = chunk_bits[:, np.newaxis, :] ^ code_bits[np.newaxis, :, :]
        dists = diffs.sum(axis=2)
        min_dists = dists.min(axis=1)
        
        valid_indices = np.where(min_dists >= 6)[0]
        
        for idx in valid_indices:
            candidates.append(chunk_words[idx])
            
        if time.time() > deadline - 5:
            break

    # 3. Select max subset from candidates
    # Greedy
    selected = []
    remaining = candidates[:]
    
    while remaining:
        w = remaining.pop(0)
        selected.append(w)
        new_remaining = []
        for r in remaining:
            if (w ^ r).bit_count() >= 6:
                new_remaining.append(r)
        remaining = new_remaining
        
    final_code = even_code + selected
    
    return final_code
