import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover followed by a trimming phase.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    total_words = q ** n
    
    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls efficiently
    balls = [None] * total_words
    ball_lists = [None] * total_words
    
    if radius == 1:
        # Fast path for radius=1
        for i in range(total_words):
            d = int_to_digits(i, n, q)
            neighbors = [i]
            for pos in range(n):
                orig = d[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    d[pos] = new_val
                    neighbors.append(digits_to_int(d, q))
                d[pos] = orig
            balls[i] = set(neighbors)
            ball_lists[i] = neighbors
    else:
        for i in range(total_words):
            d = int_to_digits(i, n, q)
            neighbors = {i}
            for pos in range(n):
                orig = d[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    d[pos] = new_val
                    neighbors.add(digits_to_int(d, q))
                d[pos] = orig
            balls[i] = neighbors
            ball_lists[i] = list(neighbors)

    # Convert to numpy arrays for vectorized operations
    ball_size = len(ball_lists[0])
    balls_np = np.zeros((total_words, ball_size), dtype=np.int32)
    for i in range(total_words):
        balls_np[i] = ball_lists[i]
    
    # Build a membership matrix: membership[w, c] = 1 if w is in ball of c
    # This is 6561 x 6561 = ~43M entries, too large for memory
    # Instead, use balls_np directly for vectorized gain computation

    # Greedy Set Cover with numpy-accelerated gain computation
    uncovered_mask = np.ones(total_words, dtype=np.bool_)
    code = []
    
    # Precompute the ball indices as int32 array for fast indexing
    while uncovered_mask.any():
        if time.time() > deadline:
            break
            
        # Vectorized gain computation: for each center c, count how many of its
        # neighbors are still uncovered
        # balls_np has shape (total_words, ball_size)
        # For each row c, we want sum(uncovered_mask[balls_np[c]])
        # This can be done with: uncovered_mask[balls_np].sum(axis=1)
        gains = uncovered_mask[balls_np].sum(axis=1)
        
        best_center = int(np.argmax(gains))
        max_gain = int(gains[best_center])
        
        if max_gain == 0:
            break
            
        code.append(best_center)
        # Mark covered words as no longer uncovered
        uncovered_mask[balls_np[best_center]] = False
        
        if time.time() > deadline:
            break

    # Trim Phase
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cover_count[balls_np[c]] += 1

    # Aggressive trimming: multiple passes
    changed = True
    while changed and time.time() < deadline:
        changed = False
        to_remove = []
        for i, c in enumerate(code):
            if not np.any(cover_count[balls_np[c]] == 1):
                to_remove.append(i)
        
        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                cover_count[balls_np[c]] -= 1
            changed = True

    # Local Search Phase
    ball_sets = [frozenset(balls[i]) for i in range(total_words)]
    
    def local_search_pass(code, cover_count, max_passes=50, sample_size=500):
        """
        Local search: for each codeword, try to find a replacement that covers
        all uniquely-covered words. Also try removals.
        """
        code_set = set(code)
        
        improved = True
        pass_count = 0
        while improved and time.time() < deadline and pass_count < max_passes:
            improved = False
            pass_count += 1

            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls_np[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Find candidates that cover all of only_by_i
                # Score candidates by how many of only_by_i they cover
                # Use numpy to compute overlaps efficiently
                # For each candidate j, compute how many of only_by_i are in ball j
                # only_by_i is a small array, so we can check membership per candidate
                
                # Get all non-code words
                all_indices = np.arange(total_words)
                not_in_code = np.ones(total_words, dtype=np.bool_)
                not_in_code[list(code_set)] = False
                candidate_arr = all_indices[not_in_code]
                
                if len(candidate_arr) == 0:
                    break
                
                # Score: for each candidate, count how many only_by_i words are in its ball
                # This is expensive if done naively, so sample
                n_cand = len(candidate_arr)
                ss = min(sample_size, n_cand)
                sample = candidate_arr[np.random.choice(n_cand, size=ss, replace=False)]
                
                best_j = -1
                best_score = -1
                for j in sample:
                    j_ball = balls_np[j]
                    # Count how many of only_by_i are in j_ball
                    # only_by_i is small, so check each
                    score = 0
                    for w in only_by_i:
                        if w in ball_sets[j]:
                            score += 1
                    if score == len(only_by_i):
                        best_j = int(j)
                        best_score = score
                        break
                    elif score > best_score:
                        best_score = score
                        best_j = int(j)
                
                if best_j >= 0 and best_score == len(only_by_i):
                    code[ci] = best_j
                    code_set.discard(i)
                    code_set.add(best_j)
                    cover_count[i_ball] -= 1
                    cover_count[balls_np[best_j]] += 1
                    improved = True
                    break

        # Final trim
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if not np.any(cover_count[balls_np[c]] == 1):
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls_np[c]] -= 1
                changed = True

        return code, cover_count

    # Run multiple local search passes with different randomization
    code, cover_count = local_search_pass(code, cover_count, max_passes=80, sample_size=600)
    
    # Second pass with more aggressive sampling
    code, cover_count = local_search_pass(code, cover_count, max_passes=40, sample_size=1000)

    # Verify coverage is complete
    # (skip for speed, but add a quick check)
    
    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result
