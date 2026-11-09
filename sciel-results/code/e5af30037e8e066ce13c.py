import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using a SEARCH approach:
    1. Multiple diverse greedy initializations (with randomization)
    2. Local search with pruning and valid swaps
    3. Keep the best result found within the time budget
    """
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]
    
    total_words = 3 ** 7
    n_words = total_words
    
    # Precompute digits and balls
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3
            
    # Precompute balls: list of lists of indices
    balls = []
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(n_words):
        ball = [i]
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers3[pos]
            base_val = i - orig_digit * p
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                j = base_val + new_digit * p
                ball.append(j)
        balls.append(ball)
    
    # Precompute ball sets for faster membership testing
    ball_sets = [set(b) for b in balls]
    
    def greedy_cover(seed):
        """Run greedy set cover with optional randomization for tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_words, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            best_idx = -1
            max_cov = 0
            best_candidates = []
            
            for i in range(n_words):
                cov = 0
                for j in balls[i]:
                    if uncovered[j]:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                    if max_cov == 15:
                        break
                elif cov == max_cov and max_cov > 0:
                    best_candidates.append(i)
            
            if best_idx == -1 and best_candidates:
                # Pick randomly from best candidates for diversity
                best_idx = rng.choice(best_candidates)
            elif best_idx == -1:
                break
                
            if best_idx == -1:
                break
                
            code.append(best_idx)
            for j in balls[best_idx]:
                uncovered[j] = False
                
        return code
    
    def local_search(code, seed):
        """Apply local search with pruning and valid swaps."""
        if time.time() - start_time > timeout:
            return code
            
        # Build coverage counts
        count = np.zeros(n_words, dtype=int)
        for c in code:
            for j in balls[c]:
                count[j] += 1
                
        # Build a set for fast lookup
        code_set = set(code)
        
        rng = random.Random(seed)
        improved = True
        
        while improved and time.time() - start_time < timeout:
            improved = False
            
            # Step A: Prune redundant codewords
            pruned = False
            new_code = []
            for c in code:
                redundant = True
                for j in balls[c]:
                    if count[j] < 2:
                        redundant = False
                        break
                if redundant:
                    for j in balls[c]:
                        count[j] -= 1
                    code_set.discard(c)
                    pruned = True
                else:
                    new_code.append(c)
            
            if pruned:
                code = new_code
                improved = True
                continue
            
            # Step B: Try swaps - use precomputed ball_sets for speed
            # Try multiple swap attempts per iteration
            for _ in range(20):
                if time.time() - start_time > timeout:
                    return code
                    
                if len(code) == 0:
                    break
                    
                c_idx = rng.randint(0, len(code) - 1)
                c = code[c_idx]
                
                # Pick a random word w
                w = rng.randint(0, n_words - 1)
                
                if w == c or w in code_set:
                    continue
                    
                # Check validity of swap c -> w using precomputed ball_sets
                ball_w_set = ball_sets[w]
                valid = True
                for j in balls[c]:
                    if j not in ball_w_set:
                        if count[j] < 2:
                            valid = False
                            break
                            
                if valid:
                    # Accept swap
                    for j in balls[c]:
                        count[j] -= 1
                    for j in balls[w]:
                        count[j] += 1
                    code[c_idx] = w
                    code_set.discard(c)
                    code_set.add(w)
                    improved = True
                    break  # Restart pruning after a swap
                    
        return code
    
    # Run multiple diverse initializations and keep the best
    best_code = None
    best_size = float('inf')
    
    num_trials = 8  # Number of diverse trials
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
            
        # Use different seeds for diversity
        seed = 42 + trial
        code = greedy_cover(seed)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
        
        # Apply local search
        code = local_search(code, seed)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
            
        # If we've found a good solution, use remaining time for more swaps
        if best_size <= 200 and time.time() - start_time < timeout - 30:
            # Do additional local search rounds on the best code
            code = local_search(list(best_code), seed + 100)
            if len(code) < best_size:
                best_code = code
                best_size = len(code)
    
    # Final check and return
    if best_code is None:
        # Fallback: return all words (invalid but safe)
        best_code = list(range(n_words))
        
    result = [list(digits[c]) for c in best_code]
    return result
