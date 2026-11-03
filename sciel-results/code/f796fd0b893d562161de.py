import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Generate all words as tuples of integers
    words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls as bitsets (Python integers)
    balls = [0] * total_words
    
    for i, w in enumerate(words):
        ball_val = 0
        # Center
        ball_val |= (1 << i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                ball_val |= (1 << j)
        balls[i] = ball_val
        
    best_code = None
    best_size = total_words + 1
    
    start_time = time.time()
    max_time = 110.0 # Leave buffer for safety
    
    while time.time() - start_time < max_time:
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            max_covered = 0
            best_candidates = []
            
            for i in range(total_words):
                covered_count = (balls[i] & uncovered).bit_count()
                
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_candidates = [i]
                elif covered_count == max_covered and covered_count > 0:
                    best_candidates.append(i)
                    
            if max_covered == 0:
                break
                
            chosen_idx = random.choice(best_candidates)
            code.append(words[chosen_idx])
            uncovered &= ~balls[chosen_idx]
            
        # Local Search: Remove redundant codewords
        # Calculate coverage counts
        counts = [0] * total_words
        for c in code:
            idx = word_to_idx[c]
            b = balls[idx]
            temp = b
            while temp:
                lsb = temp & -temp
                idx_bit = lsb.bit_length() - 1
                counts[idx_bit] += 1
                temp ^= lsb
        
        # Try to remove
        new_code = []
        for c in code:
            idx = word_to_idx[c]
            b = balls[idx]
            redundant = True
            temp = b
            while temp:
                lsb = temp & -temp
                idx_bit = lsb.bit_length() - 1
                if counts[idx_bit] == 1:
                    redundant = False
                    break
                temp ^= lsb
            
            if redundant:
                # Remove c and update counts
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    counts[idx_bit] -= 1
                    temp ^= lsb
            else:
                new_code.append(c)
        
        code = new_code

        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
            if best_size <= 181:
                break
                
    return [list(c) for c in best_code]
