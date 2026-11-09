"""Covering code K_3(8,1) via greedy set cover + local search."""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Total number of words
    total_words = q ** n
    
    # Precompute all words as a list of tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute the ball (neighbors within radius) for each word
    # For radius=1, q=3, n=8: each word covers 1 + n*(q-1) = 1 + 8*2 = 17 words
    balls = []
    for w in all_words:
        ball = {w}
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    ball.add(tuple(nw))
        balls.append(ball)
    
    # Convert to numpy arrays for faster set operations
    # ball_masks[i] is a boolean array of length total_words indicating which words are covered by word i
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i, ball in enumerate(balls):
        for w in ball:
            ball_masks[i, word_to_idx[w]] = True
    
    # Initial seed: use tetracode direct sum as starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(tuple(h + list(tail)))
    
    # Greedy set cover: start from initial, then add words that cover most uncovered
    covered = np.zeros(total_words, dtype=bool)
    for cw in initial_codewords:
        idx = word_to_idx[cw]
        covered |= ball_masks[idx]
    
    uncovered_count = total_words - np.sum(covered)
    current_code = list(initial_codewords)
    
    # If not fully covered, add words greedily
    while uncovered_count > 0 and (time.time() - start_time) < deadline - 30:
        # Find the word that covers the most uncovered words
        # For efficiency, compute coverage of uncovered for each candidate
        uncovered_mask = ~covered
        # For each word, count how many uncovered words it covers
        # This is expensive, so we sample or use a heuristic
        # Try all words but in a smart order
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            if covered[i]:
                # Skip if this word is already covered (it's in some ball, but we want to add it as codeword)
                # Actually we should consider all words as potential codewords
                pass
            cnt = np.sum(ball_masks[i] & uncovered_mask)
            if cnt > best_count:
                best_count = cnt
                best_idx = i
            if (time.time() - start_time) > deadline - 30:
                break
        if best_idx == -1 or best_count == 0:
            break
        covered |= ball_masks[best_idx]
        current_code.append(all_words[best_idx])
        uncovered_count = total_words - np.sum(covered)
    
    # Local search: try to remove redundant codewords
    improved = True
    while improved and (time.time() - start_time) < deadline - 20:
        improved = False
        new_covered = np.zeros(total_words, dtype=bool)
        # Check each codeword: is it redundant?
        for i, cw in enumerate(current_code):
            idx = word_to_idx[cw]
            # Check if removing this codeword leaves all words still covered
            # i.e., is ball[idx] a subset of the union of balls of other codewords?
            temp_covered = np.zeros(total_words, dtype=bool)
            for j, cw2 in enumerate(current_code):
                if j != i:
                    temp_covered |= ball_masks[word_to_idx[cw2]]
            if np.all(temp_covered[ball_masks[idx]]):
                # Redundant, remove it
                current_code.pop(i)
                improved = True
                break
    
    # Local search: try swapping codewords to reduce count
    # Try replacing one codeword with another that covers more while allowing removal of another
    if (time.time() - start_time) < deadline - 10:
        # Try pairwise swaps
        for i in range(len(current_code)):
            if (time.time() - start_time) > deadline - 10:
                break
            idx_i = word_to_idx[current_code[i]]
            for j in range(len(current_code)):
                if i == j:
                    continue
                idx_j = word_to_idx[current_code[j]]
                # Try replacing codeword i with a random word, see if we can remove j
                # This is complex, skip for now
                pass
    
    # Convert back to list of lists
    result = [list(cw) for cw in current_code]
    
    # Ensure no duplicates
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    return unique_result
