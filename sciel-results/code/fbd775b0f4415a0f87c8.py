import numpy as np
import itertools
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=8, radius=1.
    # General implementation for the given parameters.
    
    # Generate all words in the space
    # Using a list of tuples or integers for faster processing
    # Map word to index: index = sum(digit * q^(n-1-pos))
    
    total_words = q ** n
    
    # Precompute the "ball" of each word: indices of words within Hamming distance <= radius
    # For radius=1, ball size is 1 + n*(q-1)
    # For q=3, n=8, radius=1, size is 1 + 8*2 = 17
    
    # We represent the uncovered set as a set of indices for O(1) lookup and removal
    # Or a boolean array. A set is better for iteration over uncovered elements.
    # But we need to count how many uncovered elements a candidate covers.
    # Candidate covers a set of indices. We need intersection size with uncovered.
    
    # Strategy:
    # 1. Precompute for each index i, the list of indices in its ball.
    # 2. Maintain a set `uncovered` of indices.
    # 3. In each iteration, for each candidate i, count how many of its ball indices are in `uncovered`.
    # 4. Select i with max count.
    # 5. Remove those indices from `uncovered`.
    
    # Optimization:
    # Instead of checking all 6561 candidates every time, we can note that if a candidate's ball is fully covered, it's not useful.
    # But initially, all are useful.
    # As uncovered shrinks, many candidates become useless.
    # We can maintain a list of "active" candidates? Or just iterate all.
    # 6561 * 17 = 111k operations per iteration.
    # If we do 500 iterations, 55M operations. Python can do ~10M-50M simple ops/sec.
    # This might be close to the limit.
    # We can use NumPy for the counting if we keep `uncovered` as a boolean array.
    # `count = np.sum(uncovered[ball_indices])`
    # This is fast.
    
    # Let's use the NumPy boolean array approach but avoid the $N^2$ mask.
    # We only store the balls (list of arrays).
    
    # Precompute balls
    powers = [q ** (n - 1 - i) for i in range(n)]
    balls = []
    for i in range(total_words):
        # Convert i to word digits
        # Efficiently generate neighbors
        ball = [i]
        if radius >= 1:
            # Get digits of i
            # We can compute neighbors by adjusting i directly
            # i = d0 * q^(n-1) + ... + d_{n-1}
            # For each pos, for each new_val != old_val:
            # new_i = i + (new_val - old_val) * powers[pos]
            
            # To get old_val at pos:
            # old_val = (i // powers[pos]) % q
            
            for pos in range(n):
                p = powers[pos]
                old_val = (i // p) % q
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    n_idx = i + (new_val - old_val) * p
                    ball.append(n_idx)
        balls.append(ball)
    
    # Convert balls to numpy arrays for fast indexing
    balls_np = [np.array(b) for b in balls]
    
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    
    # To speed up, we can track the "gain" of each codeword and update it?
    # Gain[i] = number of uncovered points in ball[i].
    # When we select codeword j, we remove points from uncovered.
    # For every other codeword i, if ball[i] intersects with ball[j] (specifically, the points removed), we must decrease gain[i].
    # This is complex to maintain dynamically.
    # Given N=6561, recalculating gains from scratch is $N \times \text{ball\_size}$.
    # 6561 * 17 = 111k.
    # Using NumPy:
    # gains = np.array([np.sum(uncovered[ball]) for ball in balls_np])
    # This loop is 6561 iterations. Each is a small sum.
    # This is likely fast enough.
    
    while np.any(uncovered):
        # Calculate gains
        # List comprehension with np.sum is reasonably fast
        gains = np.array([np.sum(uncovered[ball]) for ball in balls_np])
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break # Should not happen if uncovered is not empty and balls cover the space
        
        selected.append([int((best_idx // p) % q) for p in reversed(powers)])
        
        # Update uncovered
        uncovered[balls_np[best_idx]] = False
        
    return selected
