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
            
    # Now, extend using vectorized greedy
    code = base_code[:]
    code_set = set(code)
    
    # Convert code to numpy array for vectorized distance checks
    code_arr = np.array(code, dtype=np.uint32)
    
    def bits_to_int(bits):
        """Convert a boolean numpy array of length 17 to an integer."""
        c = 0
        for i, b in enumerate(bits):
            if b:
                c |= (1 << i)
        return c
    
    def int_to_bits(c):
        """Convert integer to boolean numpy array of length 17."""
        return np.array([(c >> i) & 1 for i in range(N)], dtype=np.uint8)
    
    def vectorized_can_add(candidate_bits, code_arr):
        """Check if candidate is at distance >= MIN_DIST from all codewords."""
        # Compute XOR distances
        cand_int = 0
        for i, b in enumerate(candidate_bits):
            if b:
                cand_int |= (1 << i)
        diffs = code_arr ^ cand_int
        # Count bits
        min_dist = np.min([d.bit_count() for d in diffs]) if len(diffs) > 0 else 99
        return min_dist >= MIN_DIST
    
    def vectorized_can_add_fast(candidate_int, code_arr):
        """Fast vectorized check using numpy."""
        diffs = code_arr ^ candidate_int
        # Use lookup or bit counting
        min_dist = min(d.bit_count() for d in diffs)
        return min_dist >= MIN_DIST
    
    rng = random.Random(42)
    
    # Precompute all weight-8 and weight-9 candidates as a pool
    # This is the key: build a candidate pool and use vectorized filtering
    from itertools import combinations
    
    # Generate candidate pool: all weight 7-10 words not already in code
    candidate_pool = []
    for w in range(7, 11):
        for combo in combinations(range(N), w):
            c = 0
            for p in combo:
                c |= (1 << p)
            if c not in code_set:
                candidate_pool.append(c)
    
    # Shuffle candidates
    rng.shuffle(candidate_pool)
    
    # Vectorized greedy: add candidates that are compatible
    # Process in batches for speed
    batch_size = 5000
    added = 0
    for batch_start in range(0, len(candidate_pool), batch_size):
        if time_left() < 5.0:
            break
        batch = candidate_pool[batch_start:batch_start + batch_size]
        
        # For each candidate in batch, check against all current codewords
        # Use vectorized approach: convert code to bit matrix
        code_bits = np.array([[ (c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
        
        for cand in batch:
            if time_left() < 5.0:
                break
            if cand in code_set:
                continue
            cand_bits = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
            # XOR and count
            diffs = code_bits ^ cand_bits
            dists = diffs.sum(axis=1)
            if dists.min() >= MIN_DIST:
                code.append(cand)
                code_set.add(cand)
                added += 1
    
    # Local search: try to swap one word for another to increase total
    # 1-for-1 exchange: remove a word, see if we can add a better one
    # Actually, let's do: try adding more via random + local improvement
    
    # Second phase: more aggressive random search with vectorized checks
    for _ in range(200):
        if time_left() < 5.0:
            break
        
        # Generate random candidates
        for _ in range(2000):
            if time_left() < 5.0:
                break
            w = rng.randint(6, 12)
            positions = rng.sample(range(N), w)
            candidate = 0
            for p in positions:
                candidate |= (1 << p)
            
            if candidate in code_set:
                continue
            
            # Vectorized check
            code_bits = np.array([[ (c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
            cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
            diffs = code_bits ^ cand_bits
            dists = diffs.sum(axis=1)
            if dists.min() >= MIN_DIST:
                code.append(candidate)
                code_set.add(candidate)
    
    # Third phase: local search - try to improve by replacing pairs
    # For each word in code, try removing it and seeing if we can add 2 new words
    # This is expensive, so do it selectively
    
    # Simple local search: try random removals and re-additions
    if time_left() > 10.0:
        for attempt in range(100):
            if time_left() < 10.0:
                break
            
            # Pick a random word to remove
            idx = rng.randint(0, len(code) - 1)
            removed = code[idx]
            code.pop(idx)
            code_set.discard(removed)
            
            # Try to add back or find alternatives
            code_bits = np.array([[ (c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
            
            # Try to add several random candidates
            found = 0
            for _ in range(500):
                w = rng.randint(6, 12)
                positions = rng.sample(range(N), w)
                candidate = 0
                for p in positions:
                    candidate |= (1 << p)
                
                if candidate in code_set:
                    continue
                
                cand_bits = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
                diffs = code_bits ^ cand_bits
                dists = diffs.sum(axis=1)
                if dists.min() >= MIN_DIST:
                    code.append(candidate)
                    code_set.add(candidate)
                    code_bits = np.vstack([code_bits, cand_bits.reshape(1, -1)])
                    found += 1
                    if found >= 2:
                        break
            
            # If we didn't get back at least the removed word, restore it
            if found < 1:
                code.append(removed)
                code_set.add(removed)
    
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
