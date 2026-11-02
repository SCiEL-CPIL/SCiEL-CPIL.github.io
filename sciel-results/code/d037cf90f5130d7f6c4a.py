import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy for other cases
        total_words = q ** n
        def word_to_idx(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        def idx_to_word(idx):
            word = [0] * n
            for i in range(n - 1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        all_words = [idx_to_word(i) for i in range(total_words)]
        balls = []
        for c_idx in range(total_words):
            c_word = all_words[c_idx]
            ball = [c_idx]
            for pos in range(n):
                for val in range(q):
                    if val == c_word[pos]:
                        continue
                    new_word = c_word.copy()
                    new_word[pos] = val
                    ball.append(word_to_idx(new_word))
            balls.append(ball)
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        while True:
            if time.time() > deadline:
                break
            scores = np.array([np.sum(uncovered[b]) for b in balls], dtype=np.int32)
            best_c = np.argmax(scores)
            if scores[best_c] == 0:
                break
            code.append(all_words[best_c])
            uncovered[balls[best_c]] = False
            if not np.any(uncovered):
                break
        return code

    # Specific optimization for q=3, n=8, r=1
    total_words = 6561
    ball_size = 17
    
    # Precompute balls as numpy arrays for efficient indexing
    # balls is a list of numpy arrays, each of length 17
    balls = []
    for c_idx in range(total_words):
        c_word = []
        tmp = c_idx
        for i in range(n):
            c_word.append(tmp % q)
            tmp //= q
        c_word.reverse()
        
        ball = [c_idx]
        for pos in range(n):
            orig_val = c_word[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                # Compute index of new word
                # new_idx = c_idx + (val - orig_val) * (q ** (n - 1 - pos))
                diff = val - orig_val
                power = q ** (n - 1 - pos)
                new_idx = c_idx + diff * power
                ball.append(new_idx)
        balls.append(np.array(ball, dtype=np.int32))
    
    # Convert balls to a single 2D numpy array for faster vectorized operations
    balls_array = np.array(balls, dtype=np.int32) # shape (6561, 17)
    
    uncovered = np.ones(total_words, dtype=bool)
    code_indices = []
    
    # Greedy Set Cover
    while True:
        if time.time() > deadline:
            break
            
        # Calculate score for each candidate: number of uncovered words in its ball
        # We can use advanced indexing: uncovered[balls_array] gives a (6561, 17) bool array
        # Summing along axis 1 gives the count for each candidate.
        scores = np.sum(uncovered[balls_array], axis=1)
        
        best_c = np.argmax(scores)
        best_score = scores[best_c]
        
        if best_score == 0:
            break
            
        code_indices.append(best_c)
        
        # Update uncovered status
        uncovered[balls_array[best_c]] = False
        
        if not np.any(uncovered):
            break
            
    # Convert indices to words
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    code = [idx_to_word(i) for i in code_indices]
    
    # Optional: Local search / Pruning
    # Try to remove redundant codewords
    # If removing a codeword still leaves the space covered, remove it.
    # This is expensive to check fully, so we do a quick heuristic:
    # For each codeword, check if its unique coverage (words only it covers) is empty.
    # If yes, it's redundant.
    
    # Build a matrix: coverage[i, j] = 1 if codeword i covers word j
    # Too large (486 * 6561 ~ 3M booleans), feasible.
    
    if len(code_indices) > 0:
        # Re-verify validity and try to prune
        # Create a set of covered words for each codeword
        # We want to find if any codeword is redundant.
        
        # Efficient pruning:
        # 1. For each word, count how many codewords cover it.
        # 2. If any word is covered by exactly 1 codeword, that codeword is essential.
        # 3. If a codeword is not essential, we can try to remove it and check if validity holds.
        #    However, removing one might make another word uncovered if it was covered by 2, now 1, but that's fine.
        #    Wait, if we remove a non-essential codeword, words it covered are still covered by others.
        #    So we can just remove all non-essential codewords?
        #    No, because "essential" is defined relative to the current set.
        #    If we remove one, the counts change.
        
        # Simple iterative pruning:
        # Repeat:
        #   Count coverage for each word.
        #   Identify codewords that are the *only* cover for at least one word.
        #   If no such codewords exist, we can remove any codeword? No, that would break coverage.
        #   Actually, if every word is covered by >= 2 codewords, we can remove ANY codeword and still be valid?
        #   Yes.
        #   So:
        #   1. Compute coverage counts.
        #   2. Find words with count == 1.
        #   3. The codewords covering these words are "essential".
        #   4. Any codeword NOT in the essential set can be removed.
        #   5. Remove all non-essential codewords at once?
        #      If we remove all non-essential, the remaining essential ones still cover the words that had count==1.
        #      What about words that had count >= 2? They are still covered by at least one essential?
        #      Not necessarily. A word with count 2 might be covered by two non-essential codewords?
        #      If so, removing both would leave it uncovered.
        #      So we must be careful.
        
        # Safer approach:
        # Iterate through codewords. Try to remove one. Check validity. If valid, remove.
        # This is O(N^2 * coverage_check).
        
        # Given time constraints, the greedy result is likely close to optimal (486).
        # Pruning might save a few.
        
        # Let's try a simple pass:
        # Calculate coverage counts.
        # If min count > 1, we can remove one.
        # We can do this iteratively.
        
        # But checking validity is O(N * ball_size).
        # N ~ 486. 486 * 17 ~ 8000 ops.
        # We can try removing codewords in reverse order (last added are often less critical?).
        
        # Actually, let's just stick with the greedy result to ensure validity and speed.
        # The greedy set cover for this specific problem is known to produce ~486-500 words.
        # The best known is 486.
        
        pass

    return code
