"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a large binary code of length 18 with minimum distance 6.
    Uses forbidden-set tracking for O(1) candidate validation.
    """
    deadline = time.time() + 55.0
    N = 18
    MIN_DIST = 6
    NUM_WORDS = 2**N
    
    # Reed-Muller seed (32 words)
    def reed_muller_seed():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    seed = reed_muller_seed()
    
    def get_forbidden_words(word, N, max_dist):
        """Get all words within Hamming distance <= max_dist from word."""
        forbidden = set()
        for k in range(max_dist + 1):
            for bits_to_flip in combinations(range(N), k):
                new_word = word
                for b in bits_to_flip:
                    new_word ^= (1 << b)
                forbidden.add(new_word)
        return forbidden
    
    def greedy_with_forbidden(seed_words, rng, time_limit, order_type='random'):
        """Greedy construction with forbidden set."""
        code = list(seed_words)
        forbidden = set()
        
        for w in seed_words:
            forbidden.update(get_forbidden_words(w, N, MIN_DIST - 1))
        
        indices = np.arange(NUM_WORDS, dtype=np.int64)
        if order_type == 'random':
            rng.shuffle(indices)
        elif order_type == 'weight':
            weights = np.array([w.bit_count() for w in range(NUM_WORDS)], dtype=np.int32)
            mid = N // 2
            order = np.lexsort((indices, np.abs(weights - mid), weights))
            indices = order
        else:
            indices = np.arange(NUM_WORDS)
        
        added = 0
        for idx in range(len(indices)):
            if (idx % 200 == 0) and (time.time() > time_limit):
                break
            
            ci = int(indices[idx])
            if ci in forbidden:
                continue
            
            code.append(ci)
            added += 1
            forbidden.update(get_forbidden_words(ci, N, MIN_DIST - 1))
        
        return code
    
    best_code = seed
    best_len = len(seed)
    
    rng = np.random.default_rng(42)
    
    for order in ['random', 'weight', 'systematic']:
        if time.time() > deadline - 10:
            break
        code = greedy_with_forbidden(seed, rng, deadline - 10, order_type=order)
        if len(code) > best_len:
            best_code = code
            best_len = len(code)
    
    while time.time() < deadline - 5:
        rng2 = np.random.default_rng()
        code = greedy_with_forbidden(seed, rng2, deadline - 5, order_type='random')
        if len(code) > best_len:
            best_code = code
            best_len = len(code)
    
    return best_code

# EVOLVE-BLOCK-END
