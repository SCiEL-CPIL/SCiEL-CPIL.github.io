import time
import random
import numpy as np

def construct(q, n, radius):
    # This problem is specifically for q=3, n=7, radius=1.
    # If parameters differ, we might need a generic fallback, but the task implies fixed params.
    # We will assume q=3, n=7, radius=1 for the optimized path.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction for other cases (not expected)
        # Direct product of repetition code or similar
        # For now, return a large valid set to be safe, e.g. all words (too big)
        # Or a simple greedy for general case? 
        # Given the task contract, it passes specific q, n, radius. 
        # The seed assumes ternary length 7.
        # We will implement the specific fast path for 3,7,1.
        pass

    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precomputation for q=3, n=7
    # Total words N = 3^7 = 2187
    N = q ** n
    
    # Convert integer to tuple of digits
    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        return tuple(w)
        
    # Convert tuple of digits to integer
    def word_to_int(w):
        x = 0
        for i in range(n):
            x = x * q + w[i]
        return x

    # Precompute balls: balls[i] is a list of integers covered by codeword i
    # And neighbors: neighbors[j] is a list of codeword indices covering j
    # balls is a list of lists
    # neighbors is a list of lists
    
    # We can compute this on the fly or precompute.
    # Precomputing balls for all 2187 words is cheap.
    balls = []
    for i in range(N):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                # Create new word
                # Efficient way: 
                # The change in value at pos is (new_val - w[pos]) * q^pos
                # But since we are using base-q representation, 
                # Let's just construct the new int.
                # Actually, simpler:
                # val_change = (new_val - w[pos])
                # new_int = i + val_change * (q ** pos)
                # Wait, this works if no carry? 
                # In base q, adding delta * q^pos changes the digit at pos by delta.
                # Since we mod q, it's effectively adding delta * q^pos mod q^(pos+1)?
                # No, just:
                # new_int = i + (new_val - w[pos]) * (q ** pos)
                # Let's verify.
                # w[pos] = (i // q**pos) % q.
                # new_val = (w[pos] + delta) % q.
                # diff = new_val - w[pos].
                # If w[pos] + delta < q, diff = delta.
                # If w[pos] + delta >= q, new_val = w[pos] + delta - q, diff = delta - q.
                # If w[pos] + delta < 0, new_val = w[pos] + delta + q, diff = delta + q.
                # In all cases, i + diff * q**pos gives the correct integer representation 
                # of the new word because the other digits are unchanged.
                new_int = i + (new_val - w[pos]) * (q ** pos)
                ball.append(new_int)
        balls.append(ball)
        
    # Build neighbors
    # neighbors[j] will contain indices i such that j is in balls[i]
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        for j in balls[i]:
            neighbors[j].append(i)
            
    # Greedy Set Cover
    def greedy_cover(seed_val):
        random.seed(seed_val)
        is_covered = [False] * N
        uncovered_count = [len(balls[i]) for i in range(N)] # Initially 15
        
        codewords = []
        remaining = N
        
        # To break ties randomly, we can add a small random noise or just pick random from max
        # But standard max is fine. Randomness in tie-breaking helps.
        
        while remaining > 0:
            # Find max uncovered_count
            max_count = -1
            best_i = -1
            # We can optimize this by maintaining a heap, but N is small.
            # Linear scan is 2187 ops.
            for i in range(N):
                if uncovered_count[i] > max_count:
                    max_count = uncovered_count[i]
                    best_i = i
                    # Random tie breaking
                    if max_count == 15: # Heuristic: if full, pick random? No, just first max.
                        pass
            
            # Actually, to be more random, we can collect all i with max_count and pick one.
            # But that's O(N) extra.
            # Let's just pick the first one with max count. 
            # To introduce randomness, we can shuffle the index order once?
            # No, the state changes.
            # Let's just use the standard greedy. It's deterministic.
            # To get different results, we can randomize the initial "bias" or 
            # randomize the tie-breaking.
            
            # Let's try: if multiple have same max, pick random.
            candidates = [i for i in range(N) if uncovered_count[i] == max_count]
            if len(candidates) > 1:
                best_i = random.choice(candidates)
            else:
                best_i = candidates[0]
                
            codewords.append(best_i)
            
            # Update
            for j in balls[best_i]:
                if not is_covered[j]:
                    is_covered[j] = True
                    remaining -= 1
                    for k in neighbors[j]:
                        uncovered_count[k] -= 1
                        
            if remaining == 0:
                break
                
            # Check time
            if time.time() > deadline:
                return None # Signal timeout for this restart

        return codewords

    best_codewords = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Each restart takes ~0.05s. We can run ~1000 in 50s.
    # Let's run 200 restarts.
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        res = greedy_cover(r)
        if res is not None:
            if len(res) < best_len:
                best_len = len(res)
                best_codewords = res
                # If we hit a very good score, we might stop early?
                # Theoretical lower bound is 156.
                # If we find 160, we can try to beat it.
                # If we find 156, we are done.
                if best_len <= 160:
                    # Keep searching a bit more to ensure we don't miss 156?
                    # Or just continue to use time.
                    pass

    if best_codewords is None:
        # Fallback: return the seed result or a valid large set
        # Generate a valid set using simple greedy without restarts (deterministic)
        # Or just return the tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert integer codewords back to lists of digits
    final_words = []
    for c in best_codewords:
        final_words.append(list(int_to_word(c)))
        
    return final_words
