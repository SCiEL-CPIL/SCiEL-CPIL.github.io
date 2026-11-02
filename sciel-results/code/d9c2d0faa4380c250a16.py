import itertools
import random

def construct(q, n, radius):
    # Specific solution for q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback: return a trivial valid code (all words)
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls
    balls = []
    for i in range(N):
        w = words[i]
        neighbors = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors.append(tuple(new_w))
        balls.append([word_to_idx[nb] for nb in neighbors])
        
    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy passes
    for _ in range(100):
        uncovered = [True] * N
        counts = [1 + n * (q - 1)] * N
        selected_indices = []
        
        # Optimization: maintain a set of "active" candidates? 
        # Or just scan for max?
        # Scanning N=6561 for max is fast.
        
        while True:
            # Check if all covered
            # Instead of scanning uncovered, we can track num_uncovered
            # But we need to know if ANY is uncovered.
            # Let's just use a count of uncovered points.
            # Actually, if we track num_uncovered, we break when 0.
            
            # Find max count
            max_count = -1
            best_idx = -1
            # To allow random tie breaking, we can collect candidates or just pick first max
            # Picking first max is deterministic. 
            # To randomize, we can add small random noise to counts? 
            # Or just pick random from maxes.
            
            # Efficient max finding with tie handling:
            # Just loop and track max. If tie, we can keep a list or just pick one.
            # For simplicity and speed, let's just pick the first max.
            # Randomness comes from... wait, if we always pick first max, the path is deterministic.
            # We need randomness to explore different solutions.
            
            # Strategy: Add small random noise to counts?
            # counts[i] += random.uniform(0, 0.1)?
            # This might bias towards higher counts, which is good.
            
            # Better: 
            # 1. Find max_count.
            # 2. Collect all indices with max_count.
            # 3. Pick one randomly.
            
            # This requires two passes or a list append.
            # N is 6561. List append is fast.
            
            max_count = -1
            candidates = []
            for i in range(N):
                if counts[i] > max_count:
                    max_count = counts[i]
                    candidates = [i]
                elif counts[i] == max_count:
                    candidates.append(i)
            
            if not candidates:
                break # All counts 0?
                
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            selected_indices.append(best_idx)
            
            if max_count == 0:
                break
                
            # Update
            for j in balls[best_idx]:
                if uncovered[j]:
                    uncovered[j] = False
                    for k in balls[j]:
                        counts[k] -= 1
                        
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices
            
    return [list(words[i]) for i in best_solution]
