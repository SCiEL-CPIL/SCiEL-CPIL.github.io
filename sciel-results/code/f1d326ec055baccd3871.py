import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # GF(2^8) parameters
    PRIMITIVE_POLY = 0x11D
    
    # Precompute multiplication table for GF(2^8)
    gf_table = [[0]*256 for _ in range(256)]
    for a in range(256):
        for b in range(256):
            res = 0
            x, y = a, b
            while y:
                if y & 1:
                    res ^= x
                y >>= 1
                x <<= 1
                if x & 0x100:
                    x ^= PRIMITIVE_POLY
            gf_table[a][b] = res
    
    def gf_mul(a, b):
        return gf_table[a][b]
    
    def gf_pow(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            e >>= 1
        return res

    # Find primitive 17th root of unity in GF(2^8)
    # 255 = 15 * 17, so alpha^15 has order 17 where alpha is primitive
    alpha = gf_pow(2, 15)
    if gf_pow(alpha, 17) != 1 or alpha == 1:
        for x in range(1, 256):
            if gf_pow(x, 17) == 1 and x != 1:
                alpha = x
                break

    # The [18,9,6] extended QR code: generator polynomial has degree 9
    # For the shortened [17,9,6] code, we use the same generator but only 
    # generate codewords with the last bit = 0 (or use a parity check)
    
    # Actually, let's use a different approach: construct the code directly
    # using the known structure of the [17,9,6] code
    
    # The [17,9,6] code can be constructed from the [18,9,6] code by puncturing
    # Let's build the generator matrix directly
    
    # For the quadratic residue code of length 17 over GF(2):
    # QR(17) has generator polynomial with roots at non-residues mod 17
    # Quadratic residues mod 17: {1, 2, 4, 8, 16, 15, 13, 9}
    # Non-residues: {3, 6, 12, 7, 14, 11, 5, 10}
    
    # The generator polynomial g(x) = prod_{i in non-residues} (x - alpha^i)
    # But this gives degree 8, so [17,9,5]
    
    # For distance 6, we need the extended code. The [18,9,6] code has
    # generator polynomial of degree 9.
    
    # Let me try: use roots at ALL non-residues plus one more to get degree 9
    # Or use the fact that the [17,9,6] code is a subcode of [17,9,5]
    
    # Alternative: construct using parity-check matrix
    # H for [17,9,6] has 8 rows, each column is a nonzero vector in GF(2)^8
    # such that no 5 columns sum to zero (for d=6, no 5 columns are linearly dependent)
    
    # Let's use the known construction: the [17,9,6] code is the shortened
    # [18,9,6] extended QR code.
    
    # I'll use a greedy approach to build the code:
    # 1. Start with the zero word
    # 2. Add words that have distance >= 6 from all existing words
    
    # For efficiency, use a bitset-based approach
    
    N = 17
    MIN_DIST = 6
    
    # Precompute popcount for all 17-bit values
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Start with the zero word
    code = [0]
    
    # To speed up, we'll use a greedy algorithm with random restarts
    # First, let's try to build a large code
    
    # Precompute all pairs that have distance >= 6
    # This is too expensive (2^17 * 2^17), so we'll use a different approach
    
    # Greedy: try to add words in order of increasing weight
    # Words must have weight >= 6 (since distance from 0 must be >= 6)
    
    # Generate candidate words by weight
    from itertools import combinations
    
    candidates = []
    for w in range(MIN_DIST, N + 1):
        if time.time() > deadline - 30:
            break
        for combo in combinations(range(N), w):
            word = 0
            for pos in combo:
                word |= (1 << pos)
            candidates.append(word)
    
    # Greedy selection
    for word in candidates:
        if time.time() > deadline - 10:
            break
        if word in code:
            continue
        # Check distance to all existing codewords
        valid = True
        for existing in code:
            if popcount[word ^ existing] < MIN_DIST:
                valid = False
                break
        if valid:
            code.append(word)
    
    # Convert to list of integers
    result = list(code)
    
    return result
