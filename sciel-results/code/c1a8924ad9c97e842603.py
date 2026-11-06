import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the [17, 9, 6] Quadratic Residue Code (256 words)
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
    
    # Pre-convert generator rows to integers
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if gen_row[(j - i) % N]:
                val |= (1 << j)
        g_ints.append(val)
    
    # Generate all 2^9 codewords
    words = []
    for mask in range(2**9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        words.append(w)
    
    # Verify minimum distance for linear code
    min_w = N
    for w in words:
        if w == 0:
            continue
        weight = bin(w).count('1')
        if weight < min_w:
            min_w = weight
        if min_w < MIN_DIST:
            break
    
    if min_w < MIN_DIST:
        # Fallback to Reed-Muller based greedy
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                rm_words.append(word)
        code = list(rm_words)
        random.seed(42)
        while time.time() < deadline:
            added = False
            for _ in range(500):
                cand = random.randint(0, 2**N - 1)
                valid = all((cand ^ w).bit_count() >= MIN_DIST for w in code)
                if valid:
                    code.append(cand)
                    added = True
                    break
            if not added:
                break
        return code
    
    # We have a valid 256-word linear code. Try to extend it nonlinearly.
    code = list(words)
    
    # Strategy: Try to add words that are at distance >= 6 from all existing words.
    # Use a smarter candidate generation:
    # 1. Start with random candidates
    # 2. If stuck, try local search: flip bits of existing words and check
    
    random.seed(12345)
    
    # Precompute bit positions for faster manipulation
    bit_positions = list(range(N))
    
    # Track how many times we've failed to add a word
    fail_count = 0
    max_fails = 5000
    
    while time.time() < deadline and fail_count < max_fails:
        added = False
        
        # Try random candidates
        for _ in range(200):
            cand = random.randint(0, 2**N - 1)
            # Check distance to all codewords
            valid = True
            for w in code:
                d = (cand ^ w).bit_count()
                if d < MIN_DIST:
                    valid = False
                    break
            if valid:
                code.append(cand)
                added = True
                fail_count = 0
                break
        
        if not added:
            fail_count += 1
            
            # If we've been failing for a while, try a different strategy:
            # Take a random existing word, flip a few bits, and check
            if fail_count % 10 == 0:
                base = random.choice(code)
                # Try flipping 1-3 bits
                for _ in range(50):
                    num_flips = random.randint(1, 3)
                    flips = random.sample(bit_positions, num_flips)
                    cand = base
                    for pos in flips:
                        cand ^= (1 << pos)
                    valid = True
                    for w in code:
                        d = (cand ^ w).bit_count()
                        if d < MIN_DIST:
                            valid = False
                            break
                    if valid:
                        code.append(cand)
                        added = True
                        fail_count = 0
                        break
        
        # If we can't add any more words after many attempts, break
        if fail_count >= max_fails:
            break
    
    return code
