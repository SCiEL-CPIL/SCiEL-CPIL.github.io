"""Construct binary error-correcting code of length 18, distance 6 via greedy + local search."""
import random
import numpy as np
from itertools import combinations

N = 18
MIN_DIST = 6
MAX_WORD = 1 << N

def popcount(x):
    return bin(x).count('1')

def hamming(x, y):
    return popcount(x ^ y)

def min_dist_to_set(word, code):
    """Return minimum Hamming distance from word to any codeword in code."""
    md = N + 1
    for c in code:
        d = popcount(word ^ c)
        if d < md:
            md = d
            if md < MIN_DIST:
                break
    return md

def greedy_add(code, word):
    """Try to add word to code, return True if successful."""
    for c in code:
        if popcount(word ^ c) < MIN_DIST:
            return False
    code.append(word)
    return True

def local_search_improve(code, max_iters=200):
    """Try to add more codewords via local search: mutate random words and try to add."""
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try to add a new codeword by mutating a random base
        base = random.randint(0, (1 << N) - 1)
        # Try adding base directly
        if min_dist_to_set(base, code) >= MIN_DIST:
            code.append(base)
            improved = True
            continue
        # Try single-bit flips
        for i in range(N):
            w = base ^ (1 << i)
            if min_dist_to_set(w, code) >= MIN_DIST:
                code.append(w)
                improved = True
                break
        if improved:
            continue
        # Try double-bit flips
        for i in range(N):
            for j in range(i+1, N):
                w = base ^ (1 << i) ^ (1 << j)
                if min_dist_to_set(w, code) >= MIN_DIST:
                    code.append(w)
                    improved = True
                    break
            if improved:
                break
    return code

def greedy_construction(seed, order_type='random'):
    """Greedy construction with specified ordering."""
    rng = random.Random(seed)
    code = [0]
    
    # Generate candidates
    if order_type == 'random':
        candidates = list(range(1, 1 << N))
        rng.shuffle(candidates)
    elif order_type == 'weight_asc':
        # Sort by weight ascending
        candidates = sorted(range(1, 1 << N), key=popcount)
        # Shuffle within weights? Let's keep sorted for determinism in this pass
        # Actually, shuffling within weight buckets is better
        weights = {}
        for w in range(1, 1 << N):
            p = popcount(w)
            if p not in weights:
                weights[p] = []
            weights[p].append(w)
        candidates = []
        for p in sorted(weights.keys()):
            rng.shuffle(weights[p])
            candidates.extend(weights[p])
    elif order_type == 'weight_desc':
        weights = {}
        for w in range(1, 1 << N):
            p = popcount(w)
            if p not in weights:
                weights[p] = []
            weights[p].append(w)
        candidates = []
        for p in sorted(weights.keys(), reverse=True):
            rng.shuffle(weights[p])
            candidates.extend(weights[p])
    elif order_type == 'weight_mid':
        # Start from middle weight (9) and go out
        weights = {}
        for w in range(1, 1 << N):
            p = popcount(w)
            if p not in weights:
                weights[p] = []
            weights[p].append(w)
        
        # Order: 9, 8, 10, 7, 11, 6, 12, ...
        order = [9, 8, 10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 16, 1, 17]
        candidates = []
        for p in order:
            if p in weights:
                rng.shuffle(weights[p])
                candidates.extend(weights[p])
        
    for w in candidates:
        if min_dist_to_set(w, code) >= MIN_DIST:
            code.append(w)
            if len(code) >= 512:
                break
    return code

def refine_code(code, max_iters=100):
    """Try to add more words by iterating through a shuffled list of all words."""
    rng = random.Random(12345)
    candidates = list(range(1 << N))
    rng.shuffle(candidates)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for w in candidates:
            if min_dist_to_set(w, code) >= MIN_DIST:
                code.append(w)
                improved = True
                if len(code) >= 512:
                    return code
    return code

def construct():
    """Construct the largest possible binary code of length 18, distance 6."""
    best_code = []
    
    # Strategy 1: Greedy with different random seeds and orderings
    strategies = [
        ('random', 0), ('random', 1), ('random', 2), ('random', 3), ('random', 4),
        ('weight_asc', 0), ('weight_desc', 0), ('weight_mid', 0),
        ('weight_mid', 1), ('weight_mid', 2)
    ]
    
    for order_type, seed in strategies:
        code = greedy_construction(seed, order_type)
        code = refine_code(code, max_iters=10)
        if len(code) > len(best_code):
            best_code = code[:]
            
    # Strategy 2: Start from Reed-Muller and extend
    rm_code = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_code.append(word)
    
    # Add two more RM-like words? No, just use RM as base
    extended = list(rm_code)
    code = greedy_construction(42, 'random')
    # Combine? No, just extend RM
    rng = random.Random(42)
    candidates = list(range(1 << N))
    rng.shuffle(candidates)
    for w in candidates:
        if w in extended:
            continue
        if min_dist_to_set(w, extended) >= MIN_DIST:
            extended.append(w)
            if len(extended) >= 512:
                break
    extended = refine_code(extended, max_iters=10)
    if len(extended) > len(best_code):
        best_code = extended[:]

    # Strategy 3: Aggressive Refinement of Best
    best_code = refine_code(best_code, max_iters=50)
    
    # Strategy 4: Swap-based Local Search (if time permits)
    # Simple version: Try to swap words to free up space
    improved = True
    iters = 0
    while improved and iters < 20:
        improved = False
        iters += 1
        # Try to add a random word
        for _ in range(100):
            w = random.randint(0, (1 << N) - 1)
            if min_dist_to_set(w, best_code) >= MIN_DIST:
                best_code.append(w)
                improved = True
                break
        
        # Try swaps
        if not improved:
            for i in range(len(best_code)):
                # Try to replace best_code[i] with a random word
                # Only if the new word is "better" (hard to define, so just try)
                # We need to ensure the rest of the code is still valid
                # This is expensive: O(N * M) per swap check
                # Let's just try a few
                if i % 10 == 0: # Throttle
                    for _ in range(10):
                        new_w = random.randint(0, (1 << N) - 1)
                        # Check if new_w is far from all others except best_code[i]
                        valid = True
                        for j, c in enumerate(best_code):
                            if i == j: continue
                            if popcount(new_w ^ c) < MIN_DIST:
                                valid = False
                                break
                        if valid:
                            best_code[i] = new_w
                            improved = True
                            break
                if improved:
                    break

    # Ensure we have at least the RM code
    if len(best_code) < 32:
        best_code = rm_code

    return best_code
