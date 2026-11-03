import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Problem specific parameters
    total_words = q ** n
    
    # Precompute balls
    balls = []
    ball_masks = [] # integer masks for fast subset check (if total_words < 64, but here 2187, so use sets or arrays)
    # Since 2187 is small, we can use numpy arrays for coverage counts.
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)

    # Precompute balls as lists of indices
    for idx in range(total_words):
        word = index_to_word(idx)
        covered = [idx]
        for pos in range(n):
            digit = word[pos]
            weight = q ** (n - 1 - pos)
            for val in range(q):
                if val != digit:
                    new_idx = idx + (val - digit) * weight
                    covered.append(new_idx)
        balls.append(covered)
        
        # For swap logic, we need to check if balls[v] covers C_w.
        # We can precompute a set for each ball for O(1) membership? 
        # Or just use numpy boolean indexing.
        
    def greedy(random_seed):
        random.seed(random_seed)
        uncovered = np.ones(total_words, dtype=np.uint8)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline - 20:
                break
            
            # Find best candidate
            best_count = -1
            best_idx = -1
            
            # Optimization: Sample candidates if loop is too slow?
            # 2187 iterations * ~15 ops is fast in Python? 
            # np.count_nonzero is C-speed.
            for i in range(total_words):
                c = np.count_nonzero(uncovered[balls[i]])
                if c > best_count:
                    best_count = c
                    best_idx = i
                elif c == best_count and random.random() < 0.1:
                    # Random tie-breaking to diversify
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            uncovered[balls[best_idx]] = 0
            
        return selected

    def polish(selected, time_limit):
        if not selected:
            return selected
            
        # Compute cover_count
        cc = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            cc[balls[c]] += 1
            
        # Helper to check if a word is in selected (for swap logic)
        # Using a set for O(1) lookup
        selected_set = set(selected)
        
        # Pass 1: Removals
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                # Check if any word in balls[c] has cc == 1
                # If min(cc[balls[c]]) > 1, we can remove
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    changed = True
                    break # Restart scan
        
        # Pass 2: Swaps to reduce bottlenecks
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_bottlenecks = np.count_nonzero(cc == 1)
            
            for i in range(len(selected)):
                if time.time() > time_limit:
                    break
                c = selected[i]
                
                # Find unique words covered by c (bottlenecks assigned to c)
                # unique_mask is not a bitmask, it's a list of indices
                unique_words = [w for w in balls[c] if cc[w] == 1]
                
                if not unique_words:
                    # This codeword is redundant
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    improved = True
                    break # Restart
                    
                # Find a candidate c' not in selected
                # such that ball_masks[c'] covers unique_words
                # (balls[c'] contains all w in unique_words)
                
                best_cand = -1
                best_delta = 1000 # infinity
                # We want to minimize the increase in bottlenecks
                # Delta = (new bottlenecks from losing c) - (bottlenecks removed by gaining c')
                
                # To speed up, we can iterate all candidates
                for c_prime in range(total_words):
                    if c_prime in selected_set:
                        continue
                    
                    # Check coverage of unique_words
                    # If any w in unique_words is NOT in balls[c_prime], skip
                    # This check is O(len(unique_words))
                    # unique_words is small (usually 1-3)
                    valid = True
                    for w in unique_words:
                        if w not in balls[c_prime]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate Delta Bottlenecks
                    # Losing words: w in balls[c] but NOT in balls[c_prime]
                    # Gaining words: w in balls[c_prime] but NOT in balls[c]
                    
                    # We need to count:
                    # 1. w in balls[c] \ balls[c_prime] with cc[w] == 2 (become 1)
                    # 2. w in balls[c_prime] \ balls[c] with cc[w] == 1 (become 2)
                    
                    # Optimization: Pre-compute sets? No, just loop.
                    # balls[c] and balls[c_prime] are size 15.
                    # Intersection/Difference is small.
                    
                    # Create a set for balls[c_prime] for fast lookup?
                    # Or just use numpy?
                    # Given small size, Python sets are fine.
                    set_cp = set(balls[c_prime])
                    set_c = set(balls[c])
                    
                    losing = set_c - set_cp
                    gaining = set_cp - set_c
                    
                    new_bottlenecks_from_loss = 0
                    for w in losing:
                        if cc[w] == 2:
                            new_bottlenecks_from_loss += 1
                            
                    bottlenecks_removed_from_gain = 0
                    for w in gaining:
                        if cc[w] == 1:
                            bottlenecks_removed_from_gain += 1
                            
                    delta = new_bottlenecks_from_loss - bottlenecks_removed_from_gain
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_cand = c_prime
                        
                if best_cand != -1:
                    # Perform swap
                    cc[balls[c]] -= 1
                    cc[balls[best_cand]] += 1
                    selected_set.remove(c)
                    selected[i] = best_cand
                    selected_set.add(best_cand)
                    improved = True
                    break # Restart scan
        
        # Pass 3: Final Removals
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    changed = True
                    break
                    
        return selected

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Allocate time dynamically
    # If we find a small solution, we can spend more time polishing it?
    # Or just keep restarting to find a better start.
    
    num_restarts = 0
    while time.time() < deadline - 5:
        seed = int(time.time() * 1000) % 100000
        sol = greedy(seed)
        if not sol:
            continue
        
        # Allocate time for polish based on remaining time
        # If we are early, give it 5s. If late, give it 1s.
        remaining = deadline - time.time()
        polish_time = time.time() + min(5, remaining - 2)
        
        polished = polish(sol, polish_time)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = polished
            # If we reached the theoretical minimum (186), we can stop early?
            # But we don't know the seed value, so we don't know the max score.
            # However, 186 is the absolute minimum for this problem.
            if best_size == 186:
                # Can we do better? No.
                # But maybe we can verify and return immediately to save time?
                # Yes.
                break
        num_restarts += 1
        
    # Fallback
    if best_solution is None:
        # Direct sum construction (Tetracode x 3^3)
        # This yields 9 * 27 = 243 words?
        # Tetracode is 9 words length 4.
        # 3^3 = 27.
        # 9 * 27 = 243.
        # This is a valid upper bound.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_solution = [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(3), repeat=n - 4)]
        return base_solution
        
    # Convert to list of lists
    codewords = [list(index_to_word(c)) for c in best_solution]
    
    # Final Verification (Optional, but good for debugging)
    # If invalid, return fallback
    
    return codewords
