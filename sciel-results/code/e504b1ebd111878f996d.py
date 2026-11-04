"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np


def construct(q, n, radius):
    # New Approach: Randomized Dynamic Greedy Set Cover with Local Search
    # This differs from the static priority greedy by re-evaluating the best
    # candidate at each step and using randomization to escape local optima.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        balls = {}
        for w in all_words:
            ball = {w}
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        nw = list(w)
                        nw[i] = v
                        ball.add(tuple(nw))
            balls[w] = ball
            
        covered = set()
        codewords = []
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while len(covered) < len(all_words):
            best_c = None
            best_gain = -1
            for c in candidates:
                if c in codewords: continue
                gain = len(balls[c] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c is None: break
            codewords.append(best_c)
            covered.update(balls[best_c])
            
        return [list(c) for c in codewords]

    # Specific optimization for q=3, n=6, radius=1
    all_words = list(itertools.product(range(3), repeat=6))
    
    # Precompute coverage balls for all 729 words
    # Ball size is 1 + 6*2 = 13
    balls = {}
    for w in all_words:
        ball = {w}
        for i in range(6):
            for v in range(3):
                if v != w[i]:
                    nw = list(w)
                    nw[i] = v
                    ball.add(tuple(nw))
        balls[w] = ball

    def local_search(codewords):
        # codewords: list of tuples
        # Use coverage counts to efficiently check redundancy
        # Map word to index 0..728
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        cov_count = np.zeros(729, dtype=int)
        
        # Initialize counts
        for c in codewords:
            for w in balls[c]:
                cov_count[word_to_idx[w]] += 1
        
        # Remove redundant codewords
        # A codeword is redundant if all words in its ball are covered by at least 1 other codeword
        # i.e., cov_count[w] > 1 for all w in balls[c]
        removed = False
        while removed:
            removed = False
            for i, c in enumerate(codewords):
                is_redundant = True
                for w in balls[c]:
                    if cov_count[word_to_idx[w]] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove c
                    for w in balls[c]:
                        cov_count[word_to_idx[w]] -= 1
                    codewords.pop(i)
                    removed = True
                    break # Restart scan as indices changed
        
        return codewords

    best_solution = None
    best_len = float('inf')
    
    # Run randomized dynamic greedy multiple times
    # We use a fixed number of iterations to stay within time limits
    # 50 iterations * (729 candidates * 13 ops + local search) should be fast
    for _ in range(50):
        candidates = list(all_words)
        random.shuffle(candidates)
        
        covered = set()
        current_codewords = []
        
        # Dynamic Greedy
        while len(covered) < 729:
            best_c = None
            best_gain = -1
            
            # Optimization: Skip candidates already in current_codewords
            # Since N is small (729), linear scan is fine
            for c in candidates:
                if c in current_codewords:
                    continue
                # Calculate gain
                # balls[c] is a set of 13 tuples
                # covered is a set of tuples
                gain = len(balls[c] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c is None:
                break
            
            current_codewords.append(best_c)
            covered.update(balls[best_c])
        
        # Apply local search
        optimized = local_search(current_codewords)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized
            
        # Early termination if we hit the theoretical lower bound or known optimum
        if best_len <= 73:
            # We can try to break if we hit 72, but 73 is likely the floor
            # Let's run a few more to be sure we didn't miss 72
            if best_len <= 72:
                break

    return [list(c) for c in best_solution]

# EVOLVE-BLOCK-END
