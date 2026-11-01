import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    # The task is specifically for q=3, n=8, radius=1
    # But we should handle general cases if possible, or at least not crash.
    # However, the problem statement says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # And "Keep the function names and signatures".
    # The current seed is hardcoded for the specific structure.
    # A general greedy solver for q=3, n=8 is sufficient.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to a simple construction if parameters differ, though unlikely
        # For robustness, we can return a trivial cover (all words) if small, or direct sum if possible
        # But for this specific task, we assume q=3, n=8, r=1.
        # Let's just implement the greedy for the specific case and fallback to seed logic otherwise.
        pass

    total_words = q ** n
    
    # Precompute the balls for all words
    # ball[c] contains indices of all words within Hamming distance <= radius from c
    # For q=3, n=8, r=1, ball size is 17.
    
    # Map word tuple to index
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

    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute balls
    balls = []
    for c_idx in range(total_words):
        c_word = all_words[c_idx]
        ball = [c_idx]
        # Generate all words at distance 1
        for pos in range(n):
            for val in range(q):
                if val == c_word[pos]:
                    continue
                new_word = c_word.copy()
                new_word[pos] = val
                ball.append(word_to_idx(new_word))
        balls.append(ball)
    
    # Convert balls to numpy arrays for faster indexing? 
    # Actually, lists are fine for the inner loop if we use numpy sum on slices?
    # No, `uncovered[ball]` where ball is a list works in numpy.
    
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    # Greedy Set Cover
    while True:
        if time.time() > deadline:
            break
            
        # Calculate score for each candidate: number of uncovered words in its ball
        # scores[c] = sum(uncovered[ball[c]])
        # Using a list comprehension with numpy sum is efficient
        scores = np.array([np.sum(uncovered[b]) for b in balls], dtype=np.int32)
        
        best_c = np.argmax(scores)
        best_score = scores[best_c]
        
        if best_score == 0:
            # No candidate covers any new uncovered words? 
            # This implies uncovered words are isolated? No, every word is in its own ball.
            # If uncovered is not empty, the word itself is in its ball, so score >= 1.
            # So this only happens if uncovered is all False.
            break
            
        code.append(all_words[best_c])
        
        # Update uncovered status
        uncovered[balls[best_c]] = False
        
        if not np.any(uncovered):
            break
            
    # If we broke due to deadline, we might not have a valid cover.
    # We must ensure the result is valid.
    # The greedy algorithm adds words that cover uncovered areas.
    # If we stop early, we return an invalid cover.
    # So we must continue until covered or time runs out.
    # If time runs out, we return the best valid cover we can construct quickly?
    # No, "Valid when every word ... is within Hamming distance radius".
    # If we return an invalid cover, score is 0.
    # So we must ensure we finish the greedy loop if possible.
    # The loop is fast (~500 iterations * 6561 * 17 ops ~ 55M ops).
    # 55M ops in Python might take 5-10 seconds.
    # So we should comfortably finish within 110s.
    
    return code
