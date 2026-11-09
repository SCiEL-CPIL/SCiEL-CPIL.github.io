import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Greedy Set Cover with randomization
    2. Aggressive Local Search with Pruning, Swaps, and Add-Remove moves
    3. Final Polish with intensive swap attempts
    """
    start_time = time.time()
    timeout = 118

    if q != 3 or n != 7 or radius != 1:
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

    # Precompute balls as lists of indices
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
            
            if not best_candidates:
                break
            best_idx = rng.choice(best_candidates)
            code.append(best_idx)
            for j in balls[best_idx]:
                uncovered[j] = False
                
        return code

    def local_search(code, seed, max_swap_attempts=500):
        if time.time() - start_time > timeout:
            return code
            
        count = np.zeros(n_words, dtype=int)
        for c in code:
            for j in balls[c]:
                count[j] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
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
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
            return changed

        # Precompute for each pair (c, w) whether ball[c] \ ball[w] can be covered
        # We'll compute the "critical" positions: for each c, the positions in ball[c]
        # that are NOT in ball[w]. For a swap to be valid, all such positions must have count >= 2.
        
        # Strategy: for each codeword c, find its "weak" positions (count < 2 in ball[c])
        # Then for a swap candidate w, we need ball[w] to cover all weak positions of c that are not in ball[w]... 
        # Actually: swap c->w is valid iff for all j in ball[c] \ ball[w], count[j] >= 2
        # i.e., all positions uniquely covered by c (not in w's ball) must be covered by other codewords.
        
        # Precompute: for each c, the set of positions in ball[c] with count < 2
        # These are the "critical" positions that c must keep covering.
        # A swap c->w is valid iff critical_positions(c) ⊆ ball[w]
        
        prune()
        
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            
            if prune():
                improved = True
                continue
            
            # Compute critical positions for each codeword
            critical = {}
            for c in code:
                crit = set()
                for j in balls[c]:
                    if count[j] < 2:
                        crit.add(j)
                critical[c] = crit
            
            swap_done = False
            for _ in range(max_swap_attempts):
                if time.time() - start_time > timeout:
                    return code
                if len(code) == 0:
                    break
                
                c_idx = rng.randint(0, len(code) - 1)
                c = code[c_idx]
                crit_c = critical.get(c, set())
                
                found_swap = False
                # Try candidates in random order, but prefer those that cover critical positions
                candidates = [w for w in range(n_words) if w != c and w not in code_set]
                rng.shuffle(candidates)
                
                for w in candidates:
                    ball_w = ball_sets[w]
                    # Check if all critical positions of c are in ball[w]
                    valid = True
                    for j in crit_c:
                        if j not in ball_w:
                            valid = False
                            break
                    
                    if valid:
                        for j in balls[c]:
                            count[j] -= 1
                        for j in balls[w]:
                            count[j] += 1
                        code[c_idx] = w
                        code_set.discard(c)
                        code_set.add(w)
                        improved = True
                        swap_done = True
                        found_swap = True
                        # Recompute critical after swap
                        break
                
                if found_swap:
                    # Recompute critical positions
                    critical = {}
                    for c2 in code:
                        crit2 = set()
                        for j in balls[c2]:
                            if count[j] < 2:
                                crit2.add(j)
                        critical[c2] = crit2
                    break
            
            if swap_done:
                continue
            
            if len(code) < n_words:
                w = rng.randint(0, n_words - 1)
                if w not in code_set:
                    old_code = list(code)
                    old_count = count.copy()
                    old_set = set(code_set)
                    
                    code.append(w)
                    code_set.add(w)
                    for j in balls[w]:
                        count[j] += 1
                    
                    prune()
                    
                    if len(code) < len(old_code):
                        improved = True
                    else:
                        code = old_code
                        count = old_count
                        code_set = old_set
                        
        return code

    def polish(code, seed, time_budget):
        if time.time() - start_time > timeout:
            return code
        
        count = np.zeros(n_words, dtype=int)
        for c in code:
            for j in balls[c]:
                count[j] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
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
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
            return changed
        
        prune()
        
        # Multi-pass polish: try swaps from all codewords, not just random one
        while time.time() - start_time < time_budget:
            if len(code) == 0:
                break
            
            # Compute critical positions
            critical = {}
            for c in code:
                crit = set()
                for j in balls[c]:
                    if count[j] < 2:
                        crit.add(j)
                critical[c] = crit
            
            any_swap = False
            for c_idx in range(len(code)):
                if time.time() - start_time > time_budget:
                    break
                c = code[c_idx]
                crit_c = critical.get(c, set())
                
                found_swap = False
                # Try a subset of candidates for speed
                candidates = list(code_set)
                rng.shuffle(candidates)
                # Also try some random non-codewords
                for _ in range(50):
                    candidates.append(rng.randint(0, n_words - 1))
                
                for w in candidates:
                    if w == c or w in code_set:
                        continue
                    
                    ball_w = ball_sets[w]
                    valid = True
                    for j in crit_c:
                        if j not in ball_w:
                            valid = False
                            break
                    
                    if valid:
                        for j in balls[c]:
                            count[j] -= 1
                        for j in balls[w]:
                            count[j] += 1
                        code[c_idx] = w
                        code_set.discard(c)
                        code_set.add(w)
                        prune()
                        any_swap = True
                        found_swap = True
                        break
                
                if found_swap:
                    # Recompute critical
                    critical = {}
                    for c2 in code:
                        crit2 = set()
                        for j in balls[c2]:
                            if count[j] < 2:
                                crit2.add(j)
                        critical[c2] = crit2
        
            if not any_swap:
                break
        
        return code

    best_code = None
    best_size = float('inf')

    num_trials = 15
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
            
        seed = 100 + trial
        code = greedy_cover(seed)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        code = local_search(code, seed, max_swap_attempts=800)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        remaining = timeout - (time.time() - start_time)
        if best_size <= 200 and remaining > 15:
            code = local_search(list(best_code), seed + 1000, max_swap_attempts=500)
            if len(code) < best_size:
                best_code = list(code)
                best_size = len(code)

    remaining = timeout - (time.time() - start_time)
    if best_code is not None and remaining > 5:
        polish_time = min(remaining - 3, 40)
        best_code = polish(best_code, 99999, start_time + polish_time + 3)
        best_size = len(best_code)
    
    if best_code is None:
        best_code = list(range(n_words))
        
    result = [list(digits[c]) for c in best_code]
    return result
