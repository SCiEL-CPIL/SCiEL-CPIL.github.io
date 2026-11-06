import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Construct the [17,8,6] linear code (shortened extended QR code) - 256 words
    # Work in GF(2^8)
    def gf_mul(a, b, poly=0x1D):
        res = 0
        for i in range(8):
            if (b & 1):
                res ^= a
            b >>= 1
            if (a & 0x80):
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n, poly=0x1D):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        # Fallback: seed code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial g(x) = prod_{i in Q} (x + alpha^i)
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # Generator matrix G (9x17) for [17,9,5] QR code
    G = np.zeros((9, 17), dtype=np.uint8)
    for k in range(9):
        for j in range(9):
            bit = g_coeffs[j]
            if bit:
                pos = (j + k) % 17
                G[k, pos] = 1

    # Generate all 512 codewords of [17,9,5]
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    # Extend to [18,9,6] with parity bit
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        w_ext = np.append(w, parity)
        words_18.append(w_ext)
        
    # Shorten to [17,8,6]: keep words with 0 at position 17
    words_17_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_17_short.append(val)
    
    if len(words_17_short) != 256:
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 2. Nonlinear Extension
    words_list = list(words_17_short)
    word_set = set(words_list)
    
    # Precompute popcount for all 17-bit values (2^17 = 131072 entries)
    popcount = np.zeros(1 << 17, dtype=np.uint8)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Convert words to a NumPy array for vectorized operations
    # We will maintain this array and update it as we add words
    code_matrix = np.array([[ (w >> i) & 1 for i in range(17)] for w in words_list], dtype=np.uint8)
    
    def check_and_add_batch(candidates):
        """
        candidates: list of integers
        Returns list of candidates that are valid and adds them to words_list and code_matrix
        """
        if not candidates:
            return []
        
        # Filter out candidates already in word_set
        new_candidates = [c for c in candidates if c not in word_set]
        if not new_candidates:
            return []
        
        # Convert candidates to matrix
        cand_matrix = np.array([[ (c >> i) & 1 for i in range(17)] for c in new_candidates], dtype=np.uint8)
        
        # Compute XOR distances
        # code_matrix: (N, 17)
        # cand_matrix: (M, 17)
        # We need to compute popcount(cand_i XOR code_j) for all i, j
        # This is equivalent to summing bits of (cand_matrix[None, :, :] XOR code_matrix[:, None, :])
        
        # To save memory, we can process in chunks if M is large, but N is ~256, M ~1000
        # M*N*17 is small enough.
        
        # Expand dimensions for broadcasting
        # cand: (M, 1, 17)
        # code: (1, N, 17)
        xor_matrix = cand_matrix[:, np.newaxis, :] ^ code_matrix[np.newaxis, :, :]
        
        # Sum along the last axis to get Hamming distances
        # dist_matrix: (M, N)
        dist_matrix = np.sum(xor_matrix, axis=2)
        
        # Check if min distance for each candidate is >= 6
        min_dists = np.min(dist_matrix, axis=1)
        valid_mask = min_dists >= 6
        
        valid_candidates = [c for c, v in zip(new_candidates, valid_mask) if v]
        
        if valid_candidates:
            # Add to words_list and word_set
            for c in valid_candidates:
                words_list.append(c)
                word_set.add(c)
            
            # Update code_matrix
            new_rows = np.array([[ (c >> i) & 1 for i in range(17)] for c in valid_candidates], dtype=np.uint8)
            code_matrix = np.vstack([code_matrix, new_rows])
            
        return valid_candidates

    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Weight-ordered search (words of low weight first)
    # This is effective because the code already contains 0, and
    # low-weight words are more likely to be at distance >= 6 from all codewords.
    
    # Pass 1: Small weights
    for w in range(1, 6):
        if time.time() > deadline - 10.0:
            break
        
        # Generate all C(17, w) candidates of weight w
        candidates = []
        for comb in itertools.combinations(range(17), w):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            candidates.append(mask)
        
        # Shuffle to avoid bias
        np.random.shuffle(candidates)
        
        # Process in batches to vectorize
        batch_size = 1000
        for i in range(0, len(candidates), batch_size):
            if time.time() > deadline - 10.0:
                break
            batch = candidates[i:i+batch_size]
            check_and_add_batch(batch)
    
    # Phase 2: Vectorized Random Search
    # Use a pool of candidate weights
    # We will generate batches of random candidates and check them all at once
    
    # We need to be careful with the time. 
    # Each batch check takes O(M*N*17). 
    # Let's estimate. N=256, M=1000. 1000*256*17 = 4.3M operations.
    # NumPy should handle this in < 10ms.
    # So we can do many iterations.
    
    max_iterations = 1000
    for iteration in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
        
        batch_size = 2000
        candidates = []
        
        for _ in range(batch_size):
            # Random weight between 4 and 13
            w = random.randint(4, 13)
            if w == 0:
                cand = 0
            elif w == 17:
                cand = (1 << 17) - 1
            else:
                positions = random.sample(range(17), w)
                cand = 0
                for pos in positions:
                    cand |= (1 << pos)
            candidates.append(cand)
        
        check_and_add_batch(candidates)
        
        # If we found nothing in a large batch, we might be near the limit
        # But vectorized search is efficient, so we keep going until time is up.
        
    # Phase 3: Try complements and local variations if time remains
    if time.time() < deadline - 2.0:
        # Create a list of candidates: complements of existing words
        candidates = []
        for w in words_list:
            comp = w ^ 0x1FFFF
            candidates.append(comp)
        
        check_and_add_batch(candidates)
        
        # Local search: flip 1-3 bits of existing words
        candidates = []
        for w in words_list:
            # Flip 1 bit
            for i in range(17):
                candidates.append(w ^ (1 << i))
            # Flip 2 bits
            for i in range(17):
                for j in range(i+1, 17):
                    candidates.append(w ^ (1 << i) ^ (1 << j))
        
        # Shuffle
        np.random.shuffle(candidates)
        
        # Process in batches
        batch_size = 5000
        for i in range(0, len(candidates), batch_size):
            if time.time() > deadline - 1.0:
                break
            batch = candidates[i:i+batch_size]
            check_and_add_batch(batch)
            
    # Final return as list of integers
    return list(words_list)
