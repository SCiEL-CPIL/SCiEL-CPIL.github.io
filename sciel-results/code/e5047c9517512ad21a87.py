import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()

    # --- Step 1: Construct the [17,8,6] Linear QR Code ---
    
    PRIM_POLY = 0x11D # x^8 + x^4 + x^3 + x^2 + 1
    
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

    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
            
    if alpha is None:
        # Fallback: Greedy random construction
        return greedy_fallback()

    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Calculate generator polynomial g(x)
    # g(x) = prod_{q in Q} (x + alpha^q)
    # Coeffs in GF(2^8), but result in GF(2)
    
    g_coeffs = [1]
    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g
        
    # Convert to int
    g_int = 0
    for i in range(len(g_coeffs)):
        if g_coeffs[i] == 1:
            g_int |= (1 << i)
            
    # Generate the [17,9,6] code and filter for even weight
    code = []
    code_set = set()
    
    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
                
        # c is in [0, 2^17)
        if c.bit_count() % 2 == 0:
            code.append(c)
            code_set.add(c)
            
    # --- Step 2: Greedy Extension ---
    
    rng = random.Random(42)
    
    def can_add(candidate):
        for w in code:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
        
    # Try to add more words
    # Bias towards weights 6-11
    while time_left() > 1.0:
        # Random weight
        w = rng.randint(6, 11)
        positions = rng.sample(range(N), w)
        candidate = 0
        for p in positions:
            candidate |= (1 << p)
            
        if candidate in code_set:
            continue
            
        if can_add(candidate):
            code.append(candidate)
            code_set.add(candidate)
            
    return code

def greedy_fallback():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 17
    MIN_DIST = 6
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    
    while time.time() < deadline - 1.0:
        w = rng.randint(3, 14)
        positions = rng.sample(range(N), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
