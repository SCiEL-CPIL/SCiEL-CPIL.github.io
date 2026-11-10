import time
import itertools
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    # Precompute ball sizes: 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    num_words = q ** n
    
    # Helper functions
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    def int_to_word(w_int):
        w = []
        tmp = w_int
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        return list(reversed(w))
    
    # Precompute all balls using NumPy for speed
    # For each word, its ball contains itself + all words at distance 1
    all_words = np.arange(num_words)
    # Convert to base-q digits
    digits = np.zeros((num_words, n), dtype=np.int32)
    tmp = all_words.copy()
    for i in range(n - 1, -1, -1):
        digits[:, i] = tmp % q
        tmp //= q
    
    # Precompute ball indices
    # For efficiency, we'll compute balls on the fly but cache frequently used ones
    # Actually, let's precompute all balls since num_words=729 is small
    balls = []
    for idx in range(num_words):
        w = digits[idx]
        ball_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_word = w.copy()
                    new_word[pos] = new_val
                    new_idx = 0
                    for j in range(n):
                        new_idx = new_idx * q + new_word[j]
                    ball_set.add(new_idx)
        balls.append(sorted(ball_set))
    
    # Convert to frozensets for fast lookup
    balls_fs = [frozenset(b) for b in balls]
    
    # Step 1: Generate a good initial code using greedy set cover
    # Start with empty set, greedily add codewords that cover most uncovered words
    
    uncovered = set(range(num_words))
    code = []
    cover_count = np.zeros(num_words, dtype=np.int32)
    
    # Precompute ball as list of indices for fast iteration
    balls_list = [list(b) for b in balls]
    
    # Greedy set cover: repeatedly pick codeword covering most uncovered words
    remaining_candidates = set(range(num_words))
    
    while uncovered and time.time() < deadline - 30:
        best_idx = -1
        best_count = -1
        best_ball = None
        
        for c_idx in remaining_candidates:
            ball = balls_list[c_idx]
            # Count how many words in this ball are uncovered
            count = sum(1 for w in ball if w in uncovered)
            if count > best_count:
                best_count = count
                best_idx = c_idx
                best_ball = ball
            if count == 0:
                continue
            # Early termination if we found perfect coverage
            if count == len(uncovered):
                break
        
        if best_idx == -1 or best_count == 0:
            break
        
        # Add this codeword
        code.append(best_idx)
        remaining_candidates.discard(best_idx)
        
        # Update uncovered and cover_count
        for w in best_ball:
            if w in uncovered:
                uncovered.discard(w)
            cover_count[w] += 1
        
        # If all covered, stop
        if not uncovered:
            break
    
    # If we still have uncovered words (shouldn't happen with full greedy), add random words
    if uncovered and time.time() < deadline - 25:
        for w in list(uncovered):
            if time.time() > deadline - 25:
                break
            code.append(w)
            for w2 in balls_list[w]:
                if w2 in uncovered:
                    uncovered.discard(w2)
                cover_count[w2] += 1
    
    # Step 2: Local search - try to swap codewords to reduce total count
    # We can try removing a codeword if all its words are still covered
    # And try adding a new codeword that allows removal of multiple others
    
    code_set = set(code)
    cover_count = np.zeros(num_words, dtype=np.int32)
    for c in code:
        for w in balls_list[c]:
            cover_count[w] += 1
    
    # Greedy removal: repeatedly remove codewords that are redundant
    removed_any = True
    while removed_any and time.time() < deadline - 15:
        removed_any = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            ball = balls_list[c]
            removable = True
            for w in ball:
                if cover_count[w] <= 1:
                    removable = False
                    break
            if removable:
                # Remove this codeword
                for w in ball:
                    cover_count[w] -= 1
                code.pop(i)
                removed_any = True
    
    # Step 3: Try to improve by replacing pairs of codewords with one
    # For each pair (c1, c2), check if there exists a c3 such that
    # ball(c1) ∪ ball(c2) ⊆ ball(c3) ∪ (other covers)
    # This is complex, so let's try a simpler approach:
    # Try adding each candidate and see if we can remove more than we add
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_len = len(code)
        
        # Try each candidate not in code
        for c_new in range(num_words):
            if c_new in code_set:
                continue
            if time.time() > deadline - 10:
                break
            
            # Tentatively add c_new
            old_cover = cover_count.copy()
            for w in balls_list[c_new]:
                cover_count[w] += 1
            
            # Try to remove as many codewords as possible
            removals = []
            for c in code:
                ball = balls_list[c]
                removable = True
                for w in ball:
                    if cover_count[w] <= 1:
                        removable = False
                        break
                if removable:
                    removals.append(c)
            
            # Check if we can remove more than 1 codeword
            if len(removals) >= 2:
                # Apply removals
                new_code = [c for c in code if c not in removals]
                new_code.append(c_new)
                
                # Verify coverage
                test_cover = np.zeros(num_words, dtype=np.int32)
                for c in new_code:
                    for w in balls_list[c]:
                        test_cover[w] += 1
                if np.all(test_cover >= 1):
                    code = new_code
                    code_set = set(code)
                    cover_count = test_cover
                    improved = True
                    break
            
            # Revert
            cover_count = old_cover
    
    # Step 4: Final greedy removal pass
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            ball = balls_list[c]
            removable = True
            for w in ball:
                if cover_count[w] <= 1:
                    removable = False
                    break
            if removable:
                for w in ball:
                    cover_count[w] -= 1
                code.pop(i)
                removed_any = True
    
    # Convert to list of lists
    return [int_to_word(c) for c in code]
