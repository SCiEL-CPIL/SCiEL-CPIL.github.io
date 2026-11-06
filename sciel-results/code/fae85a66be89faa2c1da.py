import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    
    # Construct the [17, 8, 6] linear code
    # This is the shortened [18, 9, 6] quadratic residue code.
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11d
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    beta = 3
    alpha = gf_pow(beta, 15) # alpha^17 = 1
    
    a3 = gf_pow(alpha, 3)
    v = [1] * 9
    for i in range(1, 9):
        v[i] = gf_mul(v[i-1], a3)
        
    M = [[0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            M[j][i] = (v[i] >> j) & 1
            
    b = [(v[8] >> j) & 1 for j in range(8)]
    
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(8):
        pivot = -1
        for row in range(col, 8):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[col], A[pivot] = A[pivot], A[col]
        for row in range(8):
            if row != col and A[row][col] == 1:
                for k in range(col, 9):
                    A[row][k] ^= A[col][k]
                    
    c = [A[i][8] for i in range(8)]
    
    m3_poly = 0
    for i in range(8):
        if c[i]:
            m3_poly |= (1 << i)
    m3_poly |= (1 << 8)
    
    def poly_mul(x, y):
        res = 0
        while y:
            if y & 1:
                res ^= x
            y >>= 1
            x <<= 1
        return res
        
    h_poly = poly_mul(0b11, m3_poly)
    
    def poly_mod(p):
        max_bit = p.bit_length()
        for i in range(max_bit - 1, 16, -1):
            if (p >> i) & 1:
                p ^= (1 << (i - 17))
        return p & ((1 << 17) - 1)

    words = set()
    for a in range(256):
        c = poly_mod(poly_mul(a, h_poly))
        words.add(c)
        
    base_code = list(words)
    
    # Try to extend the code
    if time.time() > deadline - 10:
        return base_code
        
    # Precompute popcount for all 17-bit integers
    # 2^17 = 131072, this is fast
    popcount = np.array([i.bit_count() for i in range(1 << N)], dtype=np.uint8)
    
    # Convert base code to numpy array
    base_np = np.array(base_code, dtype=np.uint32)
    
    # Generate all possible words
    all_words = np.arange(1 << N, dtype=np.uint32)
    
    # Compute distances from all words to base code
    # This creates a (2^17, 256) array. 131072 * 256 * 4 bytes = ~134 MB.
    # This might be too large for memory or slow.
    # Let's do it in chunks or use a more efficient method.
    
    # Efficient method:
    # For each word in base_code, find words within distance < 6.
    # Mark them as invalid.
    
    valid_mask = np.ones(1 << N, dtype=bool)
    
    for c in base_np:
        # Find all words w such that popcount(w ^ c) < 6
        # This is equivalent to w being in the ball of radius 5 around c.
        # We can generate these words explicitly?
        # Number of words in ball of radius 5 is ~9400.
        # 256 * 9400 = 2.4 million operations. This is fine.
        
        # Generate all masks of weight <= 5
        # Precompute list of masks with weight <= 5
        if not hasattr(construct, '_masks'):
            masks = []
            for mask in range(1 << N):
                if mask.bit_count() <= 5:
                    masks.append(mask)
            construct._masks = np.array(masks, dtype=np.uint32)
            
        masks = construct._masks
        # Words in ball: c ^ mask
        ball_words = c ^ masks
        valid_mask[ball_words] = False
        
    # Now find valid words
    valid_indices = np.where(valid_mask)[0]
    
    # We need to select a subset of valid_indices such that pairwise distance >= 6
    # And distance to base_code is already >= 6.
    
    # Greedy selection
    new_words = []
    for w in valid_indices:
        # Check distance to new_words
        ok = True
        for nw in new_words:
            if (w ^ nw).bit_count() < MIN_DIST:
                ok = False
                break
        if ok:
            new_words.append(w)
            
    if time.time() > deadline - 2:
        return base_code
        
    final_code = base_code + new_words
    
    return final_code
