import numpy as np
import itertools
import time

def construct(q, n, radius):
    # For the specific task q=3, n=7, radius=1
    # We implement a heuristic search (Greedy Set Cover with randomization)
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction if parameters change
        # Direct product of tetracode
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Pad or truncate? The task specifies fixed n.
            # If n < 4, use trivial code
            if n <= 4:
                return [list(t) for t in itertools.product(range(3), repeat=n)]
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, r=1
    total_words = 3**7
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Precompute balls
    # Map index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % 3
            idx //= 3
        return digits

    def digits_to_index(digits):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * 3 + digits[i]
        return idx

    # Precompute all words
    words = [index_to_digits(i) for i in range(total_words)]
    
    # Precompute balls: list of lists of indices
    balls = []
    for i in range(total_words):
        w = words[i]
        ball_indices = [i]
        for pos in range(n):
            for delta in range(1, q): # 1, 2
                new_w = w.copy()
                new_w[pos] = (w[pos] + delta) % q
                ball_indices.append(digits_to_index(new_w))
        balls.append(np.array(ball_indices))

    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy iterations
    for run in range(100): # Max 100 runs
        if time.time() - start_time > time_limit:
            break
            
        cover_count = np.zeros(total_words, dtype=np.int32)
        already_covered_in_ball = np.zeros(total_words, dtype=np.int32)
        code = []
        
        while True:
            if time.time() - start_time > time_limit:
                break
                
            # Find codeword with min already_covered
            min_val = np.min(already_covered_in_ball)
            candidates = np.where(already_covered_in_ball == min_val)[0]
            
            # Random tie-breaking
            c_star = candidates[np.random.randint(len(candidates))]
            code.append(c_star)
            
            # Update counts
            b = balls[c_star]
            for w_idx in b:
                if cover_count[w_idx] == 0:
                    # This word was uncovered, now it is covered
                    # For all codewords that cover this word, increment their already_covered count
                    # We need a reverse index: which balls contain w_idx?
                    # Precomputing reverse index is expensive (2187 * 15 = 32k entries)
                    # But we can just iterate over all balls? No, that's 2187 per word.
                    # Let's precompute reverse_balls
                    pass
        
        # The update logic above is slow if done naively.
        # Let's optimize the update.
        # Precompute reverse_balls: for each word w, list of codewords c such that w in balls[c]
        # Since the graph is symmetric (distance <= 1), balls[c] contains w iff balls[w] contains c.
        # So reverse_balls[w] is just balls[w].
        
        # Let's restart the loop with this optimization in mind.
        # We will precompute reverse_balls outside the loop.
        
        break # Break out of outer loop to restructure

    # Restructure: Precompute reverse_balls
    # reverse_balls[w] = list of c such that w is in balls[c]
    # This is identical to balls[w] because d(c,w) <= 1 is symmetric.
    
    best_code = None
    best_size = float('inf')
    
    for run in range(200):
        if time.time() - start_time > time_limit:
            break
            
        cover_count = np.zeros(total_words, dtype=np.int32)
        already_covered_in_ball = np.zeros(total_words, dtype=np.int32)
        code = []
        
        while True:
            if time.time() - start_time > time_limit:
                break
                
            min_val = np.min(already_covered_in_ball)
            candidates = np.where(already_covered_in_ball == min_val)[0]
            c_star = candidates[np.random.randint(len(candidates))]
            code.append(c_star)
            
            b = balls[c_star]
            for w_idx in b:
                if cover_count[w_idx] == 0:
                    # w_idx was uncovered. Now it is covered by c_star.
                    # For every codeword c' that covers w_idx, the fact that w_idx is now covered
                    # means c' has one less "uncovered" word in its ball.
                    # Wait, already_covered_in_ball[c'] counts how many words in balls[c'] are ALREADY covered.
                    # If w_idx was uncovered (cover_count[w_idx] == 0), then it was NOT counted in already_covered_in_ball[c'].
                    # Now w_idx is covered. So for all c' covering w_idx, already_covered_in_ball[c'] should increase by 1.
                    for c_prime in balls[w_idx]:
                        already_covered_in_ball[c_prime] += 1
                cover_count[w_idx] += 1
            
            if len(code) >= best_size:
                break
            
            if not np.any(cover_count == 0):
                break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # Optional: Local search to remove redundant codewords
            # Try to remove each codeword and see if coverage is maintained
            # This is O(K * K * 15) which is fast.
            improved = True
            while improved:
                improved = False
                for i in range(len(best_code)):
                    c = best_code[i]
                    # Check if all words in balls[c] are covered by other codewords
                    # i.e., cover_count[w] > 1 for all w in balls[c]
                    # We need to recalculate cover_count or maintain it.
                    # Simpler: just check if removing c leaves any word with cover_count 0.
                    # We can simulate removal:
                    # new_cover_count = cover_count - indicator(w in balls[c])
                    # If min(new_cover_count) > 0, then c is redundant.
                    
                    # To do this efficiently, we can just check:
                    # Are all w in balls[c] covered by at least one other codeword?
                    # i.e., is there any w in balls[c] such that cover_count[w] == 1 AND c is the only one covering it?
                    # Actually, if cover_count[w] == 1, it must be covered by c (since c is in code).
                    # So if any w in balls[c] has cover_count[w] == 1, then c is essential?
                    # No, if cover_count[w] == 1, it means exactly one codeword covers it.
                    # If that codeword is c, then removing c makes w uncovered.
                    # So c is redundant iff for all w in balls[c], cover_count[w] > 1.
                    
                    redundant = True
                    for w_idx in balls[c]:
                        if cover_count[w_idx] == 1:
                            redundant = False
                            break
                    
                    if redundant:
                        # Remove c
                        for w_idx in balls[c]:
                            cover_count[w_idx] -= 1
                        best_code.pop(i)
                        improved = True
                        break # Restart loop as indices changed
                
                if len(best_code) < best_size:
                    best_size = len(best_code)

    if best_code is None:
        # Fallback
        return [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Convert indices back to lists of digits
    return [words[c] for c in best_code]
