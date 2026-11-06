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
    
    # Track best code size for local search
    best_code_size = len(code)
    
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
        added_any = False
        for candidate in batch_candidates:
            if can_add_vec(candidate):
                code.append(candidate)
                code_set.add(candidate)
                code_arr = np.append(code_arr, np.array([candidate], dtype=np.uint32))
                added_any = True
        
        if added_any:
            best_code_size = len(code)
        
        if time_left() < 2.0:
            break
    
    # Local search: 1-for-1 swap to find nonlinear extensions
    # Try removing one codeword and adding two new ones
    if time_left() > 5.0:
        local_rng = random.Random(123)
        
        # Precompute distances for faster swaps
        def can_add_with_exclusion(candidate, exclude_idx):
            """Check if candidate can be added when exclude_idx is removed."""
            if candidate in code_set:
                return False
            # Check against all codewords except exclude_idx
            for i in range(len(code)):
                if i == exclude_idx:
                    continue
                if (candidate ^ code[i]).bit_count() < MIN_DIST:
                    return False
            return True
        
        improved = True
        while improved and time_left() > 3.0:
            improved = False
            n = len(code)
            if n < 2:
                break
            
            # Try random 1-for-1 swaps
            for _ in range(min(1000, max(100, n * 2))):
                if time_left() < 3.0:
                    break
                
                # Pick a codeword to remove
                remove_idx = local_rng.randint(0, n - 1)
                removed = code[remove_idx]
                
                # Try to find two candidates that can replace it
                found_pair = False
                for _ in range(50):
                    if time_left() < 3.0:
                        break
                    
                    # Generate first candidate
                    w1 = local_rng.randint(6, 11)
                    pos1 = local_rng.sample(range(N), w1)
                    cand1 = 0
                    for p in pos1:
                        cand1 |= (1 << p)
                    
                    if cand1 == removed or cand1 in code_set:
                        continue
                    
                    # Check if cand1 can be added with remove_idx excluded
                    ok1 = True
                    for i in range(n):
                        if i == remove_idx:
                            continue
                        if (cand1 ^ code[i]).bit_count() < MIN_DIST:
                            ok1 = False
                            break
                    
                    if not ok1:
                        continue
                    
                    # Generate second candidate
                    w2 = local_rng.randint(6, 11)
                    pos2 = local_rng.sample(range(N), w2)
                    cand2 = 0
                    for p in pos2:
                        cand2 |= (1 << p)
                    
                    if cand2 == removed or cand2 in code_set or cand2 == cand1:
                        continue
                    
                    # Check if cand2 can be added with remove_idx excluded and cand1 present
                    ok2 = True
                    for i in range(n):
                        if i == remove_idx:
                            continue
                        if (cand2 ^ code[i]).bit_count() < MIN_DIST:
                            ok2 = False
                            break
                    if (cand2 ^ cand1).bit_count() < MIN_DIST:
                        ok2 = False
                    
                    if ok2:
                        # Perform the swap
                        code[remove_idx] = cand1
                        code_set.discard(removed)
                        code_set.add(cand1)
                        
                        # Add cand2
                        code.append(cand2)
                        code_set.add(cand2)
                        
                        code_arr = np.array(code, dtype=np.uint32)
                        improved = True
                        found_pair = True
                        break
                
                if found_pair:
                    break
        
        if len(code) > best_code_size:
            best_code_size = len(code)
    
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
