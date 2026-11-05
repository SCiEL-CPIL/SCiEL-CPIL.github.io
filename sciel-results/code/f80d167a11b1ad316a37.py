import time
import random

# EVOLVE-BLOCK-START
N = 18
MIN_DIST = 6

def hamming_distance(a, b):
    return (a ^ b).bit_count()

def reed_muller_construction():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    Every affine Boolean function on F_2^4 is a word: 32 words, distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def greedy_random_construction(seed=42, max_iter=2000000, deadline=None):
    """Greedy random construction to add words to the RM base code."""
    rng = random.Random(seed)
    words = reed_muller_construction()
    word_set = set(words)
    
    # Pre-compute bit masks for fast weight calculation? No, bit_count is fast.
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Generate random word
        # Bias towards low weights
        weight = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9])
        positions = rng.sample(range(N), min(weight, N))
        word = 0
        for p in positions:
            word |= 1 << p
        
        if word in word_set:
            continue
        
        valid = True
        for w in words:
            if hamming_distance(word, w) < MIN_DIST:
                valid = False
                break
        if valid:
            words.append(word)
            word_set.add(word)
    
    return words

def construct():
    """
    Attempts QR code, falls back to robust greedy search.
    """
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    # Try QR Code first (fast)
    try:
        g_coeffs = [1, 1, 0, 1, 1, 0, 1, 1, 1]
        G_17 = []
        for i in range(9):
            row = 0
            for j in range(9):
                bit = g_coeffs[j]
                pos = (i + j) % 17
                if bit:
                    row |= (1 << pos)
            G_17.append(row)
            
        G_18 = []
        for row in G_17:
            parity = row.bit_count() % 2
            row_18 = row | (parity << 17)
            G_18.append(row_18)
            
        words = []
        for mask in range(512):
            word = 0
            for i in range(9):
                if (mask >> i) & 1:
                    word ^= G_18[i]
            words.append(word)
            
        # Verify min distance
        min_d = 18
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                d = hamming_distance(words[i], words[j])
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        break
            if min_d < MIN_DIST:
                break
                
        if min_d >= MIN_DIST:
            return words
    except Exception:
        pass

    # Fallback: Greedy Search
    best_words = []
    seeds = [42, 123, 999, 2023, 777]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        current_words = greedy_random_construction(seed=seed, max_iter=1000000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
            
    # If best is still small, try one more long run
    if len(best_words) < 100 and time.time() < deadline:
        current_words = greedy_random_construction(seed=42, max_iter=2000000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
            
    return best_words

# EVOLVE-BLOCK-END
