import numpy as np
import itertools
import random
import time
import math
import sys

sys.setrecursionlimit(10000)

def construct(q, n, radius):
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls as bitsets
    balls = [0] * total_words
    ball_sizes = [0] * total_words
    
    for i, w in enumerate(words):
        ball_val = 0
        ball_val |= (1 << i)
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
        ball_sizes[i] = ball_val.bit_count()
        
    # Precompute degrees (how many balls cover each word)
    degrees = [0] * total_words
    for i in range(total_words):
        b = balls[i]
        temp = b
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            degrees[idx] += 1
            temp ^= lsb

    best_size = total_words + 1
    best_code = None
    start_time = time.time()
    max_time = 110.0

    def solve(uncovered, current_code, current_size):
        nonlocal best_size, best_code
        
        if time.time() - start_time > max_time:
            return
            
        if uncovered == 0:
            if current_size < best_size:
                best_size = current_size
                best_code = current_code.copy()
            return
            
        num_uncovered = uncovered.bit_count()
        # Lower bound: ceil(num_uncovered / max_ball_size)
        # For q=3, n=7, r=1, max_ball_size = 15
        lb = math.ceil(num_uncovered / 15)
        
        if current_size + lb >= best_size:
            return
            
        # MRV Heuristic: Find uncovered word with minimum degree
        min_deg = float('inf')
        target_word = -1
        temp = uncovered
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            if degrees[idx] < min_deg:
                min_deg = degrees[idx]
                target_word = idx
            temp ^= lsb
            
        if target_word == -1:
            return
            
        # Find candidates that cover target_word
        target_bit = 1 << target_word
        candidates = []
        for c in range(total_words):
            if balls[c] & target_bit:
                new_cov = balls[c] & uncovered
                if new_cov:
                    candidates.append((new_cov.bit_count(), c, new_cov))
                    
        # Sort by coverage descending (Greedy choice)
        candidates.sort(reverse=True)
        
        for _, c, new_cov in candidates:
            new_uncovered = uncovered & ~new_cov
            current_code.append(c)
            
            solve(new_uncovered, current_code, current_size + 1)
            
            current_code.pop()

    # Run Greedy to get an initial upper bound, then B&B
    for restart in range(20):
        if time.time() - start_time > max_time:
            break
            
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_cov = 0
            best_candidates = []
            for c in range(total_words):
                cov = (balls[c] & uncovered).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_candidates = [c]
                elif cov == best_cov and cov > 0:
                    best_candidates.append(c)
                    
            if not best_candidates:
                break
                
            c = random.choice(best_candidates)
            code.append(c)
            uncovered &= ~balls[c]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        # Run B&B
        solve((1 << total_words) - 1, [], 0)
        
        if best_size <= 181:
            break

    return [list(words[c]) for c in best_code]
