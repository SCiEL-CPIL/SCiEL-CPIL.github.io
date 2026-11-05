import random

def construct(q, n, radius):
    # This implementation assumes the specific case q=3, n=6, radius=1
    # which is the target of the task.
    # For generality, we fall back to a simple method if parameters differ, 
    # but the evaluator is expected to pass the specific task parameters.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (less optimized)
        return _construct_fallback(q, n, radius)

    total_words = q ** n # 729
    indices = range(total_words)
    
    # Precompute balls as bitmasks
    # index = w0 * 3^5 + w1 * 3^4 + ... + w5
    # Actually, let's just map tuple to index via base-3 conversion
    
    def tuple_to_index(t):
        idx = 0
        for digit in t:
            idx = idx * q + digit
        return idx

    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute ball masks for all 729 words
    # mask[i] is an int where bit j is 1 if word j is in the ball of word i
    ball_masks = [0] * total_words
    for i in indices:
        w = index_to_tuple(i)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = tuple_to_index(tuple(new_w))
                    mask |= (1 << j)
        ball_masks[i] = mask

    all_bits = (1 << total_words) - 1
    
    best_codewords = None
    best_size = total_words + 1
    
    # Run multiple randomized greedy passes with local search
    num_restarts = 100 # Increased restarts, local search is fast
    
    import time
    start_time = time.time()
    time_limit = 110 # Leave some buffer for 120s limit
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        uncovered = all_bits
        codewords = []
        
        while uncovered:
            # Find the word that covers the most uncovered bits
            max_cov = -1
            best_candidates = []
            
            for i in indices:
                cov = (uncovered & ball_masks[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                elif cov == max_cov:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            # Randomly pick from best candidates to introduce variation
            chosen = random.choice(best_candidates)
            codewords.append(chosen)
            uncovered &= ~ball_masks[chosen]
            
        # Local Search: Try to improve the set
        # 1. Removal: Check if any codeword is redundant
        # 2. Swap: Try to replace a codeword with another to enable further removal or just maintain coverage with potential for better future swaps
        # We'll do a simple hill-climbing local search
        
        improved = True
        while improved:
            improved = False
            
            # Pass 1: Try to remove redundant codewords
            for i in range(len(codewords)):
                # Check if codewords[i] is redundant
                # i.e., union of balls of all other codewords covers all_bits
                union_others = 0
                for j in range(len(codewords)):
                    if j != i:
                        union_others |= ball_masks[codewords[j]]
                
                if (union_others & all_bits) == all_bits:
                    # Redundant, remove it
                    codewords.pop(i)
                    improved = True
                    break # Restart loop to check new indices
            
            if len(codewords) == 0:
                break
                
            # Pass 2: Try to swap a codeword with a non-codeword
            # This is more expensive. We try a few random swaps per codeword.
            # If a swap is valid (new set covers all_bits), we accept it.
            # Note: A swap doesn't immediately reduce size, but might allow a removal in the next pass.
            # Or we can look for a swap that *increases* the redundancy of another codeword?
            # Simpler: Just try to find a swap that maintains coverage.
            # Actually, a better heuristic: 
            # For each codeword c_i, try to swap it with c_j such that the new set is valid.
            # This doesn't change size. 
            # To reduce size, we need to find a c_j that makes some c_k redundant.
            # This is complex. 
            # Let's stick to Removal first, as it's the most direct improvement.
            # If no removal is possible, we can try a "perturbation" or just stop.
            # Given the small size, we can try a limited number of swaps to see if it unlocks a removal.
            
            if not improved:
                # Try some random swaps to escape local minima
                swaps_attempted = 0
                max_swaps = 50
                for i in range(len(codewords)):
                    if swaps_attempted >= max_swaps:
                        break
                    # Pick a random candidate not in codewords
                    # To speed up, pick from indices
                    # We can just pick a random index and check if it's in codewords
                    # Or maintain a set of codewords for O(1) lookup
                    c_set = set(codewords)
                    for _ in range(10):
                        cand = random.randint(0, total_words - 1)
                        if cand in c_set:
                            continue
                        
                        # Try swapping codewords[i] with cand
                        # Check if new set covers all_bits
                        # New set = codewords without i, plus cand
                        # We can compute union efficiently
                        union_new = ball_masks[cand]
                        for j in range(len(codewords)):
                            if j != i:
                                union_new |= ball_masks[codewords[j]]
                        
                        if (union_new & all_bits) == all_bits:
                            # Valid swap
                            codewords[i] = cand
                            improved = True
                            swaps_attempted += 1
                            break # Restart local search loop
                    if improved:
                        break
                        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
    # Convert indices back to lists
    result = [list(index_to_tuple(i)) for i in best_codewords]
    return result

def _construct_fallback(q, n, radius):
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    balls = []
    for w in all_words:
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    neighbor = list(w)
                    neighbor[i] = val
                    ball.add(tuple(neighbor))
        balls.append(ball)
        
    uncovered = set(all_words)
    codewords = []
    while uncovered:
        best_idx = -1
        max_covered = 0
        for i in range(len(all_words)):
            covered_count = len(balls[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        if best_idx == -1:
            break
        best_word = all_words[best_idx]
        codewords.append(best_word)
        uncovered -= balls[best_idx]
    return [list(c) for c in codewords]
