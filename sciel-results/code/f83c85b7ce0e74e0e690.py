import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # GF(2^8) arithmetic
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
        
        # Check even weight
        if c.bit_count() % 2 == 0:
            base_code.append(c)
            
    code = base_code[:]
    code_set = set(code)
    
    rng = random.Random(42)
    
    # Helper: check if a candidate can be added using vectorized ops
    def can_add(cand, code_arr):
        """Check if cand has min distance >= MIN_DIST from all codewords in code_arr (numpy array)."""
        cand_bits = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
        diffs = code_arr ^ cand_bits
        dists = diffs.sum(axis=1)
        return dists.min() >= MIN_DIST
    
    # Phase 1: Greedy addition with structured candidates
    # Precompute all weight-6 to weight-11 candidates not in code
    candidate_pool = []
    for w in range(6, 12):
        for combo in combinations(range(N), w):
            c = 0
            for p in combo:
                c |= (1 << p)
            if c not in code_set:
                candidate_pool.append(c)
    
    rng.shuffle(candidate_pool)
    
    # Build code bit matrix
    code_arr = np.array([[(c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
    
    # Greedy: try to add candidates
    for cand in candidate_pool:
        if time_left() < 5.0:
            break
        if cand in code_set:
            continue
        if can_add(cand, code_arr):
            code.append(cand)
            code_set.add(cand)
            # Update code_arr by appending new row
            new_row = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
            code_arr = np.vstack([code_arr, new_row])
    
    # Phase 2: Random greedy with diverse weights
    for _ in range(500):
        if time_left() < 5.0:
            break
        w = rng.randint(6, 11)
        positions = rng.sample(range(N), w)
        candidate = 0
        for p in positions:
            candidate |= (1 << p)
        if candidate in code_set:
            continue
        if can_add(candidate, code_arr):
            code.append(candidate)
            code_set.add(candidate)
            new_row = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
            code_arr = np.vstack([code_arr, new_row])
    
    # Phase 3: 1-for-1 swap local search (remove one, try to add two)
    # This can escape local optima by finding nonlinear extensions
    improved = True
    swap_iterations = 0
    while improved and time_left() > 10.0 and swap_iterations < 200:
        improved = False
        swap_iterations += 1
        
        # Try each codeword as a removal candidate
        for i in range(len(code)):
            if time_left() < 10.0:
                break
            removed = code[i]
            # Build code without this element
            code_without = code[:i] + code[i+1:]
            code_without_set = set(code_without)
            code_without_arr = np.array([[(c >> i2) & 1 for i2 in range(N)] for c in code_without], dtype=np.uint8)
            
            # Try to add two new words
            added_count = 0
            # Generate candidates biased toward the removed word's neighborhood
            for attempt in range(200):
                if time_left() < 10.0:
                    break
                # Mix: some random, some near removed word
                if attempt % 3 == 0:
                    # Near removed word: flip 2-4 bits
                    flipped = rng.sample(range(N), rng.randint(2, 4))
                    candidate = removed
                    for p in flipped:
                        candidate ^= (1 << p)
                else:
                    w = rng.randint(6, 11)
                    positions = rng.sample(range(N), w)
                    candidate = 0
                    for p in positions:
                        candidate |= (1 << p)
                
                if candidate in code_without_set:
                    continue
                if can_add(candidate, code_without_arr):
                    code_without.append(candidate)
                    code_without_set.add(candidate)
                    new_row = np.array([(candidate >> i2) & 1 for i2 in range(N)], dtype=np.uint8)
                    code_without_arr = np.vstack([code_without_arr, new_row])
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count >= 2:
                # Accept the swap
                code = code_without
                code_set = code_without_set
                code_arr = code_without_arr
                improved = True
                break
    
    # Phase 4: Final greedy pass with remaining budget
    for _ in range(2000):
        if time_left() < 3.0:
            break
        w = rng.randint(6, 12)
        positions = rng.sample(range(N), w)
        candidate = 0
        for p in positions:
            candidate |= (1 << p)
        if candidate in code_set:
            continue
        if can_add(candidate, code_arr):
            code.append(candidate)
            code_set.add(candidate)
            new_row = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
            code_arr = np.vstack([code_arr, new_row])
    
    return code

def greedy_construct(deadline):
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
