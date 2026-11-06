import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for general case (not expected to be called for this task)
        return _construct_general(q, n, radius, deadline)

    total_words = 2187
    full_mask = (1 << total_words) - 1
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * 3 + d
        return idx

    def idx_to_word(idx):
        w = [0] * 7
        for i in range(6, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute balls (coverage masks)
    # balls[i] is a bitmask of all words within distance 1 of word i
    balls = [0] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        for pos in range(7):
            for new_digit in range(3):
                if new_digit == w[pos]:
                    continue
                nw = w[:]
                nw[pos] = new_digit
                j = word_to_idx(nw)
                mask |= (1 << j)
        balls[i] = mask

    # Helper to run greedy + local search
    def solve(seed_val):
        random.seed(seed_val)
        uncovered = full_mask
        code = []
        code_set = set()
        
        # Greedy Phase
        while uncovered:
            best_idx = -1
            max_covered = 0
            candidates = []
            
            # Iterate through all possible codewords
            # Optimization: Skip those already in code_set
            for c in range(total_words):
                if c in code_set:
                    continue
                covered = (balls[c] & uncovered).bit_count()
                if covered > max_covered:
                    max_covered = covered
                    best_idx = c
                    candidates = [c]
                elif covered == max_covered and max_covered > 0:
                    candidates.append(c)
            
            if best_idx == -1:
                break
                
            # Randomized tie-breaking
            chosen = random.choice(candidates) if len(candidates) > 1 else best_idx
            
            code.append(chosen)
            code_set.add(chosen)
            uncovered &= ~balls[chosen]
            
            if time.time() > deadline - 10:
                break

        # Local Search Phase
        # 1. Removal of redundant codewords
        # 2. Swap attempts to reduce size further (if possible via complex swaps, but simple swaps keep size)
        # Actually, to reduce size, we need to find a codeword that can be removed.
        # The greedy phase already tries to cover everything.
        # The "Removal" phase in the original code checks if a codeword is critical.
        
        # Compute coverage counts
        coverage = [0] * total_words
        for c in code:
            mask = balls[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                coverage[idx] += 1
                mask ^= lsb

        # Iterative Removal
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each codeword
            # Order: Random or fixed? Random helps escape local minima.
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(code): continue
                c = code[i]
                can_remove = True
                mask = balls[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if coverage[idx] == 1:
                        can_remove = False
                        break
                    mask ^= lsb
                
                if can_remove:
                    # Remove c
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        coverage[idx] -= 1
                        mask ^= lsb
                    
                    code.pop(i)
                    code_set.discard(c)
                    improved = True
                    if len(code) <= 186:
                        return code

        # Swap Phase: Try to replace a codeword with another to allow removal?
        # Or simply try to find a smaller code via random restarts in the main loop.
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple instances with different seeds
    # 1. Deterministic (seed 0)
    # 2. Randomized seeds
    
    num_restarts = 0
    while time.time() < deadline - 20:
        seed_val = num_restarts
        code = solve(seed_val)
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
            if best_size <= 186:
                break
        
        num_restarts += 1
        # If we are stuck at a size, increase restart frequency or vary seed
        if num_restarts % 5 == 0 and time.time() < deadline - 30:
             # Maybe try a different greedy strategy? 
             # For now, just more random restarts.
             pass

    if best_code is None:
        # Fallback: Return a valid code (e.g. all words, or a large greedy code)
        # But solve() should always return a valid code.
        best_code = solve(0)

    return [idx_to_word(c) for c in best_code]

def _construct_general(q, n, radius, deadline):
    # Simple fallback
    total_words = q ** n
    # Greedy
    full_mask = (1 << total_words) - 1
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    balls = [0] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        for pos in range(n):
            for new_digit in range(q):
                if new_digit == w[pos]: continue
                nw = w[:]
                nw[pos] = new_digit
                j = word_to_idx(nw)
                mask |= (1 << j)
        balls[i] = mask
    uncovered = full_mask
    code = []
    code_set = set()
    while uncovered:
        best_idx = -1
        max_covered = 0
        for c in range(total_words):
            if c in code_set: continue
            covered = (balls[c] & uncovered).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_idx = c
        if best_idx == -1: break
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered &= ~balls[best_idx]
        if time.time() > deadline: break
    return [idx_to_word(c) for c in code]
