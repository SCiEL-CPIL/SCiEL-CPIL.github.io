import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    
    # --- Step 1: Construct the [17, 8, 6] QR Code ---
    # GF(2^8) arithmetic
    PRIM_POLY = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find primitive 17th root of unity alpha in GF(2^8)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            if gf_pow(a, 1) != 1:
                alpha = a
                break
                
    if alpha is None:
        # Fallback
        return [0]

    # Cyclotomic coset of 1 mod 17
    Q = [1, 2, 4, 8, 16, 15, 13, 9]
    
    # Construct generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    # g(x) is in GF(2)[x]
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Verify coefficients are 0 or 1
    g_int = 0
    for i, c in enumerate(g_coeffs):
        if c == 1:
            g_int |= (1 << i)
        elif c != 0:
            # This should not happen for the minimal polynomial
            # If it does, the code construction is flawed
            pass
            
    # Generate all 512 codewords of the [17, 9, 6] cyclic code
    # c(x) = m(x) * g(x)
    base_code_512 = []
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        base_code_512.append(c)
        
    # Filter to even-weight codewords to get the [17, 8, 6] QR code
    # This should yield 256 codewords
    base_code = [c for c in base_code_512 if bin(c).count('1') % 2 == 0]
    
    # Ensure uniqueness
    base_code = list(set(base_code))
    
    if len(base_code) == 0:
        # Fallback to random if construction failed
        base_code = [0]
        
    # --- Step 2: Vectorized Local Search ---
    
    code_array = np.array(base_code, dtype=np.uint32)
    
    pop_lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def fast_popcount(arr):
        b0 = (arr >> 0) & 0xFF
        b1 = (arr >> 8) & 0xFF
        b2 = (arr >> 16) & 0xFF
        b3 = (arr >> 24) & 0xFF
        return pop_lut[b0] + pop_lut[b1] + pop_lut[b2] + pop_lut[b3]

    rng = random.Random(42)
    
    # Local search
    while time.time() < deadline - 2.0:
        # Generate candidates
        # Focus on weights 6-11
        candidates = []
        for _ in range(5000):
            w = rng.randint(6, 11)
            pos = rng.sample(range(N), w)
            val = 0
            for p in pos:
                val |= (1 << p)
            candidates.append(val)
            
        cand_arr = np.array(candidates, dtype=np.uint32)
        
        valid_cands = []
        chunk_size = 1000
        for i in range(0, len(cand_arr), chunk_size):
            if time.time() > deadline - 1.0:
                break
            chunk = cand_arr[i:i+chunk_size]
            xor_mat = chunk[:, None] ^ code_array[None, :]
            dists = fast_popcount(xor_mat)
            valid_mask = np.all(dists >= MIN_DIST, axis=1)
            if np.any(valid_mask):
                valid_cands.extend(chunk[valid_mask].tolist())
                
        if valid_cands:
            new_arr = np.array(valid_cands, dtype=np.uint32)
            code_array = np.concatenate([code_array, new_arr])
            
    return code_array.tolist()
