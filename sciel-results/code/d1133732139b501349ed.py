import time
import random

# EVOLVE-BLOCK-START
N = 18
MIN_DIST = 6
SIZE = 1 << N

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

# Precompute all error patterns with weight < MIN_DIST
ERROR_PATTERNS = []
def generate_error_patterns():
    global ERROR_PATTERNS
    ERROR_PATTERNS = []
    # Weight 0
    ERROR_PATTERNS.append(0)
    # Weight 1
    for i in range(N):
        ERROR_PATTERNS.append(1 << i)
    # Weight 2
    for i in range(N):
        for j in range(i+1, N):
            ERROR_PATTERNS.append((1 << i) | (1 << j))
    # Weight 3
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                ERROR_PATTERNS.append((1 << i) | (1 << j) | (1 << k))
    # Weight 4
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    ERROR_PATTERNS.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
    # Weight 5
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    for m in range(l+1, N):
                        ERROR_PATTERNS.append((1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m))

generate_error_patterns()

def mark_forbidden(forbidden, word):
    """Mark all words at distance < MIN_DIST from 'word' as forbidden."""
    for e in ERROR_PATTERNS:
        forbidden[word ^ e] = 1

def greedy_construction_fast(base_words, deadline, seed):
    """Greedy search using a forbidden array."""
    rng = random.Random(seed)
    words = list(base_words)
    
    # Initialize forbidden array
    forbidden = bytearray(SIZE)
    
    # Mark all base words as forbidden (and their neighbors)
    for w in words:
        mark_forbidden(forbidden, w)
        
    # Greedy loop
    while time.time() < deadline:
        # Generate random candidate
        candidate = rng.randint(0, SIZE - 1)
        
        if forbidden[candidate]:
            continue
            
        # It's valid!
        words.append(candidate)
        mark_forbidden(forbidden, candidate)
        
    return words

def construct():
    """
    Constructs a binary code of length 18 with minimum distance 6.
    Uses Reed-Muller base code and then fast greedy search.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for 90s limit
    
    # Base code
    base_words = reed_muller_construction()
    best_words = base_words
    
    # Try several seeds
    seeds = [42, 123, 999, 2023, 777, 1234, 5678, 9012, 3456, 7890]
    
    time_per_seed = 5.0
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Start from base code
        current_words = greedy_construction_fast(base_words, deadline, seed)
        
        if len(current_words) > len(best_words):
            best_words = current_words

    return best_words

# EVOLVE-BLOCK-END
