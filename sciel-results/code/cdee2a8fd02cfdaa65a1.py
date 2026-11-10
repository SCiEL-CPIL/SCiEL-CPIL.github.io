import itertools
import numpy as np
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general inputs if possible, or assume the task's specific parameters.
    # The prompt says: "the evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle general q, n, radius.
    
    if radius == 0:
        return [list(map(int, word)) for word in itertools.product(range(q), repeat=n)]
        
    # If the space is small, we can use a general greedy set cover.
    # If the space is large, we might need heuristics.
    # For n=6, q=3, size is 729.
    # For larger n, we might fall back to the seed or a simpler heuristic.
    
    total_words = q ** n
    if total_words > 50000:
        # Fallback to seed-like construction or simple random if too large
        # But the task is specifically n=6, q=3.
        # Let's stick to the greedy for small sizes.
        pass

    # Convert words to integers for efficient indexing
    # word_to_int: tuple of n ints -> int
    # int_to_word: int -> tuple of n ints
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of ints within radius of i
    
    # To be efficient, we can generate neighbors on the fly or precompute.
    # Precomputation for 729 words * 13 neighbors is trivial.
    
    # Helper to convert int to base-q tuple
    def int_to_word(x):
        return [x // (q ** i) % q for i in range(n)]
        
    def word_to_int(w):
        x = 0
        for i in range(n):
            x += w[i] * (q ** i)
        return x

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i)
        # Center
        neighbors[i].append(i)
        # Radius 1 neighbors
        for pos in range(n):
            for delta in range(1, q):
                new_w = w.copy()
                new_w[pos] = (w[pos] + delta) % q
                # Also check (w[pos] - delta) % q? 
                # The loop `for delta in range(1, q)` covers all other values.
                # Wait, if q=3, delta=1,2. 
                # (w[pos]+1)%3 and (w[pos]+2)%3 cover the other two values.
                # So this is correct.
                neighbors[i].append(word_to_int(new_w))
                
    # Greedy Set Cover
    # We want to minimize the number of codewords.
    
    best_code = None
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Run multiple random restarts
    for restart in range(100):
        uncovered = [True] * total_words
        code = []
        
        while any(uncovered):
            # Find the word that covers the most uncovered neighbors
            max_count = -1
            best_candidate = -1
            candidates = []
            
            for i in range(total_words):
                count = 0
                for nb in neighbors[i]:
                    if uncovered[nb]:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_candidate = i
                    candidates = [i]
                elif count == max_count and count > 0:
                    candidates.append(i)
                    
            if max_count == 0:
                # Should not happen if uncovered is not empty
                break
                
            # Random tie-breaking
            chosen = candidates[restart % len(candidates)] if len(candidates) > 1 else best_candidate
            # Better: random.choice(candidates)
            import random
            random.seed(restart) # Deterministic per restart? Or just use random
            chosen = random.choice(candidates)
            
            code.append(chosen)
            
            # Mark neighbors as covered
            for nb in neighbors[chosen]:
                uncovered[nb] = False
                
        if not best_code or len(code) < len(best_code):
            best_code = code
            
        if len(best_code) <= 73: # Known best
            break
            
        if time.time() - start_time > time_limit:
            break
            
    if not best_code:
        # Fallback to seed if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert back to list of lists of ints
    return [int_to_word(c) for c in best_code]
