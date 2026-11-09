import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Parameters from task
    # q=3, n=6, radius=1
    N = q ** n
    deadline = time.time() + 110.0  # 10s margin
    
    # Precompute balls for all words in {0..q-1}^n
    # Represent words as indices 0..N-1
    # Convert index to base-q digits
    def idx_to_digits(idx):
        return [int(idx // (q ** i) % q) for i in range(n)]
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (q ** i)
        return idx

    balls = []
    for idx in range(N):
        digits = idx_to_digits(idx)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                new_digits = digits[:]
                new_digits[pos] = (digits[pos] + delta) % q
                ball.append(digits_to_idx(new_digits))
        balls.append(ball)
        
    # Precompute ball sets for fast membership testing
    ball_sets = [set(b) for b in balls]

    # Initial code: Tetracode direct sum
    # Tetracode words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(digits_to_idx(word))
            
    codewords = initial_codewords[:]
    cov = np.zeros(N, dtype=np.int32)
    
    for c in codewords:
        cov[balls[c]] += 1
        
    # Local Search
    while time.time() < deadline:
        # 1. Greedy Deletion Pass
        i = 0
        progress = False
        while i < len(codewords):
            c = codewords[i]
            # Check if any word in ball(c) has coverage 1
            # If min coverage in ball is > 1, c is redundant
            if np.min(cov[balls[c]]) > 1:
                cov[balls[c]] -= 1
                codewords.pop(i)
                progress = True
                # Do not increment i, check new word at i
            else:
                i += 1
                
        if progress:
            continue
            
        # 2. Swap Pass (Random Perturbation)
        # Try to swap a random codeword with a random word
        # This helps escape local minima where no single deletion is possible,
        # but a move might enable future deletions.
        
        # To be efficient, we can try a few swaps per loop
        # or just one. Let's try one.
        
        if len(codewords) == 0:
            break
            
        i = random.randint(0, len(codewords) - 1)
        c_old = codewords[i]
        c_new = random.randint(0, N - 1)
        
        if c_new == c_old:
            continue
            
        # Tentative removal
        cov[balls[c_old]] -= 1
        
        # Find uncovered words
        missing = np.where(cov < 1)[0]
        
        if len(missing) == 0:
            # c_old was redundant (shouldn't happen if deletion pass was thorough, 
            # but maybe due to race conditions or logic gaps? 
            # Actually, if min cov was 1, it's essential. So missing should be non-empty.)
            # If missing is empty, it means c_old was NOT essential. 
            # We should have deleted it. 
            # Let's just add c_new and accept.
            cov[balls[c_new]] += 1
            codewords[i] = c_new
        else:
            # Check if c_new covers all missing words
            # missing is an array of indices
            # Check if all are in ball_sets[c_new]
            b_set = ball_sets[c_new]
            valid_swap = True
            for m in missing:
                if m not in b_set:
                    valid_swap = False
                    break
                    
            if valid_swap:
                cov[balls[c_new]] += 1
                codewords[i] = c_new
            else:
                # Revert
                cov[balls[c_old]] += 1
                
    # Convert indices back to list of lists
    result = [idx_to_digits(c) for c in codewords]
    return result
