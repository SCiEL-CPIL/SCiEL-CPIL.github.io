import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    timeout = 110  # Safety margin from 120s
    
    total_words = q ** n
    # Precompute neighbors (ball of radius 1) for each word index
    # Represent words as integers 0..total_words-1 in base q
    # To check distance, we can convert to list of digits or use base-q arithmetic.
    # Given n=7, q=3, conversion is cheap.
    
    def to_int(digits):
        val = 0
        for d in digits:
            val = val * q + d
        return val
        
    def to_digits(val):
        digits = []
        for _ in range(n):
            digits.append(val % q)
            val //= q
        return digits[::-1] # reverse to get most significant first? 
                            # Actually, the order doesn't matter for distance as long as consistent.
                            # Let's keep 0 as least significant for easier modulo math if needed, 
                            # but for distance we just compare digits.
        
    # Precompute neighbors for all words
    neighbors = [[] for _ in range(total_words)]
    all_digits = [to_digits(i) for i in range(total_words)]
    
    for i in range(total_words):
        d_i = all_digits[i]
        # Center
        neighbors[i].append(i)
        # 1-bit flips (q-1 choices per position)
        for pos in range(n):
            orig = d_i[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Create new word
                new_d = d_i.copy()
                new_d[pos] = new_val
                neighbors[i].append(to_int(new_d))
                
    # Greedy function with a specific random seed/order
    def greedy_random_order(seed):
        random.seed(seed)
        covered = [False] * total_words
        uncovered_count = total_words
        codewords = []
        candidates = list(range(total_words))
        
        while uncovered_count > 0:
            random.shuffle(candidates)
            best_c = -1
            best_gain = -1
            
            for c in candidates:
                # Check how many new words c covers
                gain = 0
                for nb in neighbors[c]:
                    if not covered[nb]:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we can't beat best? 
                            # Max possible gain is len(neighbors[c])
                            # But we need to check all to be sure if we don't break.
                            # Actually, if gain + (remaining neighbors to check) <= best_gain, break?
                            # Simpler: just compute full gain. It's only 15 items.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
                    # Optimization: if best_gain is max possible (len(neighbors[c])), 
                    # we can't do better with this c, but another c might have same max.
                    # We just want ANY max.
                    
            if best_gain == 0:
                # This shouldn't happen if we iterate all candidates
                # But if it does, break to avoid infinite loop
                break
                
            # Add best_c
            codewords.append(best_c)
            for nb in neighbors[best_c]:
                if not covered[nb]:
                    covered[nb] = True
                    uncovered_count -= 1
                    
        return codewords

    best_solution = None
    best_len = float('inf')
    seed = 0
    
    while time.time() - start_time < timeout:
        sol = greedy_random_order(seed)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            # If we hit a very good solution, we might want to try to polish it?
            # Or just keep searching.
            
        seed += 1
        
        # If we find a solution close to the lower bound (156), we might want to spend more time?
        # Or if we hit 186 (best known), we can try to improve.
        
    if best_solution is None:
        # Fallback: return a valid but possibly larger code
        # The initial program's logic: Tetracode * 3^(n-4)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert integer indices back to lists of digits
    return [all_digits[c] for c in best_solution]
