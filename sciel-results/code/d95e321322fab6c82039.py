"""Covering code K_3(6,1) - greedy set cover with local optimization."""
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Build the universe: all words in {0,1,2}^6
    # Each word is represented as an integer 0..728
    total = q ** n
    words = list(range(total))
    
    # Precompute: for each word, its ball (itself + all words at distance <= radius)
    # A word at integer index w has digits via base-q representation
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return w
    
    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx
    
    # Precompute balls for all words
    balls = {}
    for w_idx in range(total):
        w = idx_to_word(w_idx)
        ball = [w_idx]
        for pos in range(n):
            for delta in range(1, q):
                new_w = w[:]
                new_w[pos] = (w[pos] + delta) % q
                ball.append(word_to_idx(new_w))
        balls[w_idx] = set(ball)
    
    # Greedy set cover: start with all words as potential codewords
    # Use a priority-based greedy: repeatedly pick the word covering most uncovered
    
    # Initial: use the direct sum construction as starting point
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Build initial codeword set from tetracode x all tails
    import itertools
    initial_cw = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n-4):
            cw = h + list(tail)
            initial_cw.append(word_to_idx(cw))
    
    # Track coverage
    covered = set()
    for cw in initial_cw:
        covered.update(balls[cw])
    
    codewords = set(initial_cw)
    
    # Greedy removal: try to remove codewords that don't leave gaps
    # A codeword can be removed if all words it uniquely covers are also covered by others
    # More precisely: for each codeword, check if removing it leaves any word uncovered
    
    def is_redundant(cw_idx, codeword_set):
        """Check if removing cw_idx leaves all its ball covered by remaining codewords."""
        for w in balls[cw_idx]:
            # Check if w is covered by some other codeword
            found = False
            for other in codeword_set:
                if other != cw_idx and w in balls[other]:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Iterative removal pass
    improved = True
    while improved:
        improved = False
        for cw in list(codewords):
            if time.time() > deadline:
                break
            if is_redundant(cw, codewords):
                codewords.remove(cw)
                improved = True
        if time.time() > deadline:
            break
    
    # Local optimization: try swaps - replace one codeword with another to enable more removals
    # Also try: for each codeword, try replacing it with a random word and see if we can then remove more
    
    best_size = len(codewords)
    best_set = set(codewords)
    
    # Simulated annealing style: try random swaps
    temperature = 1.0
    cooling = 0.999
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        current_size = len(codewords)
        
        # Try a swap: pick a random codeword and replace with a random other word
        cw_list = list(codewords)
        if not cw_list:
            break
        
        old_cw = random.choice(cw_list)
        new_cw = random.randint(0, total - 1)
        
        if new_cw in codewords:
            continue
        
        # Tentatively swap
        new_set = set(codewords)
        new_set.discard(old_cw)
        new_set.add(new_cw)
        
        # Check coverage
        covered_check = set()
        for cw in new_set:
            covered_check.update(balls[cw])
        
        if len(covered_check) == total:
            # Valid swap - now try to remove redundant codewords
            temp_set = set(new_set)
            removed_any = True
            while removed_any:
                removed_any = False
                for cw in list(temp_set):
                    if is_redundant(cw, temp_set):
                        temp_set.remove(cw)
                        removed_any = True
            
            if len(temp_set) < best_size:
                best_size = len(temp_set)
                best_set = set(temp_set)
                codewords = set(temp_set)
            elif len(temp_set) == current_size:
                # Accept with some probability (annealing)
                if random.random() < temperature:
                    codewords = set(temp_set)
        
        temperature *= cooling
    
    # Final cleanup: one more removal pass
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            for cw in list(codewords):
                if is_redundant(cw, codewords):
                    codewords.remove(cw)
                    improved = True
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_set = set(codewords)
    
    # Use the best set found
    final_cw_set = best_set if len(best_set) <= len(codewords) else codewords
    
    # Convert back to list of words
    result = [idx_to_word(cw) for cw in final_cw_set]
    
    # Verify validity
    all_covered = set()
    for cw in result:
        cw_idx = word_to_idx(cw)
        all_covered.update(balls[cw_idx])
    
    if len(all_covered) != total:
        # Fallback to initial construction
        result = [idx_to_word(cw) for cw in initial_cw]
    
    return result
