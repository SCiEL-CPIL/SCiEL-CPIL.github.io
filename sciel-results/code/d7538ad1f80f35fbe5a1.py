import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
        
    # 1. Construct the [17,8,6] QR code
    # GF(2^8) arithmetic
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 = 0x11D
    PRIM_POLY = 0x11D
    
    def gf_mul(a, b):
        res = 0
        for i in range(8):
            if (b >> i) & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, p):
        res = 1
        for _ in range(p):
            res = gf_mul(res, a)
        return res

    # Find alpha (primitive element)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break
            
    if alpha is None:
        # Fallback to random greedy if construction fails
        return greedy_construct(deadline)

    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Calculate generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    g_coeffs = [1]
    
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Convert to binary polynomial coefficients
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    while len(bin_g) < 9:
        bin_g.append(0)
        
    # Convert bin_g to integer for fast shifting
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)
            
    # Generate all 512 codewords of the [17,9,6] code
    base_code = []
    
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        
        # Reduce mod x^17 - 1
        # deg(m) <= 8, deg(g) = 8. deg(prod) <= 16.
        # So c is already in [0, 2^17).
        
        # Check even weight
        if c.bit_count() % 2 == 0:
            base_code.append(c)
            
    # Now, extend using greedy with vectorized distance checks
    code = base_code[:]
    code_set = set(code)
    
    # Precompute code as numpy array for vectorized operations
    code_arr = np.array(code, dtype=np.uint32)
    
    # Bit mask for each position
    bit_masks = np.array([1 << i for i in range(N)], dtype=np.uint32)
    
    # Precompute popcount lookup for 8-bit chunks
    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def popcount_vec(arr):
        """Vectorized popcount for uint32 array using lookup table."""
        # Split into 4 bytes
        b0 = arr & 0xFF
        b1 = (arr >> 8) & 0xFF
        b2 = (arr >> 16) & 0xFF
        b3 = (arr >> 24) & 0xFF
        return popcount_table[b0] + popcount_table[b1] + popcount_table[b2] + popcount_table[b3]
    
    def can_add_vec(candidate):
        """Vectorized check if candidate can be added."""
        # Compute XOR with all codewords
        xors = code_arr ^ candidate
        # Compute popcount
        dists = popcount_vec(xors)
        return np.all(dists >= MIN_DIST)
    
    rng = random.Random(42)
    
    # Batch greedy: generate many candidates, check them all at once
    BATCH_SIZE = 2000
    
    while time_left() > 2.0:
        # Generate a batch of random candidates
        batch_candidates = []
        for _ in range(BATCH_SIZE):
            w = rng.randint(6, 11)
            positions = rng.sample(range(N), w)
            candidate = 0
            for p in positions:
                candidate |= (1 << p)
            if candidate not in code_set:
                batch_candidates.append(candidate)
        
        if not batch_candidates:
            continue
        
        # Check each candidate
        for candidate in batch_candidates:
            if can_add_vec(candidate):
                code.append(candidate)
                code_set.add(candidate)
                code_arr = np.append(code_arr, np.array([candidate], dtype=np.uint32))
        
        if time_left() < 2.0:
            break
    
    return code

def greedy_construct(deadline):
    # Fallback if QR construction fails
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
