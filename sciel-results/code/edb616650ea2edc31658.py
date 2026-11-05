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

    # 2. Nonlinear Extension with O(1) distance lookups via popcount table
    words_list = list(words_17_short)
    word_set = set(words_list)
    
    # Precompute popcount for all 17-bit values (2^17 = 131072 entries)
    # This enables O(1) Hamming distance computation: popcount[a ^ b]
    popcount = np.zeros(1 << 17, dtype=np.uint8)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    n_words = len(words_list)
    
    def is_valid(candidate, current_words):
        """Check if candidate has distance >= 6 from all words in current_words."""
        for w in current_words:
            if popcount[candidate ^ w] < 6:
                return False
        return True
    
    def add_word(candidate):
        """Add a word to the code."""
        words_list.append(candidate)
        word_set.add(candidate)
    
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
        # For w <= 5, C(17,w) <= 6188, which is manageable
        candidates = []
        for comb in itertools.combinations(range(17), w):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            candidates.append(mask)
        
        # Shuffle to avoid bias
        np.random.shuffle(candidates)
        
        # Try each candidate
        for cand in candidates:
            if time.time() > deadline - 10.0:
                break
            if cand not in word_set and is_valid(cand, words_list):
                add_word(cand)
    
    # Pass 2: Random search with diverse weights
    # Use a pool of candidate weights
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a batch of random candidates
        batch_size = 500
        found_in_batch = 0
        
        for _ in range(batch_size):
            if time.time() > deadline - 5.0:
                break
            
            # Random weight between 4 and 13 (avoid very low/high weights)
            w = random.randint(4, 13)
            # Generate random word of weight w
            if w == 0:
                cand = 0
            elif w == 17:
                cand = (1 << 17) - 1
            else:
                positions = random.sample(range(17), w)
                cand = 0
                for pos in positions:
                    cand |= (1 << pos)
            
            if cand in word_set:
                continue
            
            if is_valid(cand, words_list):
                add_word(cand)
                found_in_batch += 1
        
        # If we found nothing in a large batch, we might be near the limit
        if found_in_batch == 0 and len(words_list) > 260:
            # Try a different strategy: local search
            # Pick a random existing word and try flipping 1-3 bits
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                
                base = random.choice(words_list)
                n_flips = random.randint(1, 3)
                flip_positions = random.sample(range(17), n_flips)
                cand = base
                for pos in flip_positions:
                    cand ^= (1 << pos)
                
                if cand in word_set:
                    continue
                
                if is_valid(cand, words_list):
                    add_word(cand)
                    found_in_batch += 1
    
    # Phase 3: Vectorized local search / hill climbing
    # Try to find more words by checking batches of candidates at once using NumPy
    # This is faster than checking one by one when many candidates are invalid
    
    if time.time() < deadline - 2.0:
        # Create a matrix of current words for vectorized distance checks
        # We'll check candidates in batches
        
        # Try to add complements of existing words
        for w in list(words_list):
            if time.time() > deadline - 2.0:
                break
            comp = w ^ 0x1FFFF  # 17-bit complement
            if comp not in word_set:
                if is_valid(comp, words_list):
                    add_word(comp)
    
    # Phase 4: Final random search with more iterations
    if time.time() < deadline - 1.0:
        # Continue random search until time runs out
        attempts = 0
        max_attempts = 100000
        while time.time() < deadline - 0.5 and attempts < max_attempts:
            # Random weight between 3 and 14
            w = random.randint(3, 14)
            if w == 0:
                cand = 0
            elif w == 17:
                cand = (1 << 17) - 1
            else:
                positions = random.sample(range(17), w)
                cand = 0
                for pos in positions:
                    cand |= (1 << pos)
            
            if cand not in word_set:
                if is_valid(cand, words_list):
                    add_word(cand)
            attempts += 1
    
    # Final return as list of integers
    return list(words_list)
