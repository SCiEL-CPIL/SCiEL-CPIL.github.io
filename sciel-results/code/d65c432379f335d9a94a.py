import random
import time
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    # The evaluator kills after 90s wall-clock. We use a safe margin of 80s.
    time_limit = 80.0

    def hamming_dist(a, b):
        return bin(a ^ b).count('1')

    def is_valid(c, codewords):
        for c2 in codewords:
            if hamming_dist(c, c2) < MIN_DIST:
                return False
        return True

    def try_add_random(codewords, attempts=200):
        for _ in range(attempts):
            w = random.getrandbits(N)
            if is_valid(w, codewords):
                codewords.append(w)
                return True
        return False

    def greedy_add(codewords, max_adds=50):
        added = 0
        while added < max_adds:
            if not try_add_random(codewords, attempts=100):
                break
            added += 1
            if time.time() - start_time > time_limit:
                break

    def local_search(codewords, max_iters=5000):
        # Simple local search: try to swap one codeword for two, or just add
        # This is a heuristic.
        # A more robust approach for this specific problem size is to run multiple
        # random greedy restarts and keep the best.
        # However, "swap" logic can be effective.
        # Let's try a simpler "kick" and re-greedy approach if stuck.
        
        best_set = codewords[:]
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Try to add a random word
            if try_add_random(best_set, attempts=50):
                continue # Success, keep going
            
            # If stuck, try a "kick": remove a random word, perturb it, and re-greedy
            # Or just remove a random word and try to fill the space.
            if len(best_set) > 0:
                idx = random.randint(0, len(best_set) - 1)
                removed = best_set.pop(idx)
                
                # Try to add 2 new words to replace the 1 removed
                # This increases size by 1 if successful.
                found = 0
                for _ in range(100):
                    if found >= 2:
                        break
                    w = random.getrandbits(N)
                    if is_valid(w, best_set):
                        best_set.append(w)
                        found += 1
                
                # If we didn't add enough to compensate or improve, 
                # we might have reduced size or kept it same.
                # We only keep this change if size increased? 
                # Actually, if we removed 1 and added 2, size +1.
                # If we removed 1 and added 1, size 0.
                # If we removed 1 and added 0, size -1.
                
                # We want to maximize size. 
                # If current size < best_size, revert?
                # This simple loop doesn't track best_size easily without backtracking.
                # Let's stick to a simpler strategy: Random Restart Greedy.
                # It is often more robust for small N.
                # But we are already in a local search context.
                
                # Let's just break out of this specific local search variant
                # and rely on the outer loop's restarts.
                # Or, revert if size decreased.
                if len(best_set) < len(codewords):
                    best_set.append(removed)
                    # Sort to keep consistent? Not necessary.
                elif len(best_set) == len(codewords):
                    # No gain, maybe revert to avoid drifting?
                    # Let's keep it, it might be in a different basin.
                    pass
                else:
                    # Improved!
                    codewords = best_set[:]
                    
        return codewords

    # Strategy:
    # 1. Start with a good seed (Reed-Muller or random)
    # 2. Greedy add
    # 3. Local search
    # 4. Repeat with different seeds if time permits.
    
    best_codewords = []
    
    # Run multiple independent trials
    num_trials = 10
    trial_time_limit = time_limit / num_trials
    
    for t in range(num_trials):
        trial_start = time.time()
        
        # Initialize with a random codeword
        codewords = [random.getrandbits(N)]
        
        # Greedy phase
        greedy_add(codewords, max_adds=100)
        
        # Local search phase
        # The local_search function above is a bit messy. 
        # Let's use a cleaner "random restart" approach for the main loop, 
        # and a simpler improvement step.
        
        # Improvement: Try to add more.
        # If stuck, try "swap" logic properly.
        
        # A better local search for packing:
        # 1. Try to add.
        # 2. If fail, pick a random codeword c.
        # 3. Remove c.
        # 4. Try to add 2 random codewords.
        # 5. If successful (size +1), keep. Else revert.
        
        improved = True
        while improved and (time.time() - trial_start) < trial_time_limit * 0.7:
            improved = False
            
            # Try to add
            if try_add_random(codewords, attempts=20):
                improved = True
                continue
                
            # Try swap: remove 1, add 2
            if len(codewords) > 0:
                idx = random.randint(0, len(codewords) - 1)
                c_removed = codewords[idx]
                codewords.pop(idx)
                
                added_count = 0
                new_words = []
                for _ in range(50):
                    w = random.getrandbits(N)
                    if is_valid(w, codewords + new_words):
                        new_words.append(w)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count == 2:
                    codewords.extend(new_words)
                    improved = True
                elif added_count == 1:
                    # Size same. Might be useful to change the set composition?
                    # Usually not, unless it opens up space for later.
                    # Let's revert to be safe and deterministic in size tracking.
                    codewords.append(c_removed)
                else:
                    codewords.append(c_removed)
        
        if len(codewords) > len(best_codewords):
            best_codewords = codewords[:]
            
        if time.time() - start_time > time_limit:
            break

    # Final verification
    # Ensure distinctness and distance
    valid_codewords = []
    seen = set()
    for c in best_codewords:
        if c in seen:
            continue
        seen.add(c)
        # Check against all previously accepted
        ok = True
        for c2 in valid_codewords:
            if hamming_dist(c, c2) < MIN_DIST:
                ok = False
                break
        if ok:
            valid_codewords.append(c)
            
    return valid_codewords
