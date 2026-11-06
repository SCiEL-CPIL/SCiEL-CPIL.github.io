import sys
import time
import random

def construct():
    n = 4096
    max_sum = 3 * n
    
    start_time = time.time()
    timeout = 55.0
    
    best_set = []
    best_size = 0
    
    def try_greedy(seed, randomize):
        nonlocal best_set, best_size, start_time
        
        if time.time() - start_time > timeout:
            return
        
        random.seed(seed)
        
        used_sums = bytearray(max_sum + 1)
        pair_sums = set()
        chosen = []
        
        current_max = 0
        
        while True:
            if time.time() - start_time > timeout:
                return
            
            valid_candidates = []
            
            for x in range(current_max + 1, n + 1):
                if time.time() - start_time > timeout:
                    return
                
                # 1. Check 3x
                if used_sums[3 * x]:
                    continue
                
                # 2. Check 2x in pair_sums
                if 2 * x in pair_sums:
                    continue
                
                # 3. Check a+2x and a+x
                valid_so_far = True
                for a in chosen:
                    if used_sums[a + 2 * x]:
                        valid_so_far = False
                        break
                    if a + x in pair_sums:
                        valid_so_far = False
                        break
                
                if not valid_so_far:
                    continue
                
                # 4. Check a+b+x
                for i in range(len(chosen)):
                    a = chosen[i]
                    for j in range(i, len(chosen)):
                        b = chosen[j]
                        if used_sums[a + b + x]:
                            valid_so_far = False
                            break
                    if not valid_so_far:
                        break
                
                if valid_so_far:
                    valid_candidates.append(x)
                    if len(valid_candidates) >= 5:
                        break
            
            if not valid_candidates:
                break
            
            if randomize:
                x = random.choice(valid_candidates)
            else:
                x = valid_candidates[0]
            
            # Add x
            old_chosen = chosen[:]
            chosen.append(x)
            current_max = x
            
            # Update pair_sums
            pair_sums.add(2 * x)
            for a in old_chosen:
                pair_sums.add(a + x)
                
            # Update used_sums
            used_sums[3 * x] = 1
            for a in old_chosen:
                used_sums[a + 2 * x] = 1
            for i in range(len(old_chosen)):
                a = old_chosen[i]
                for j in range(i, len(old_chosen)):
                    b = old_chosen[j]
                    used_sums[a + b + x] = 1
            
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = chosen[:]
                
        return

    # Run a mix of greedy and randomized restarts
    # Greedy is fast and reliable. Randomized might find better.
    # Let's run 20 pure greedy (seed 0-19) and 20 randomized (seed 20-39)
    
    for i in range(20):
        try_greedy(i, False)
        if time.time() - start_time > timeout:
            break
            
    for i in range(20):
        try_greedy(100 + i, True)
        if time.time() - start_time > timeout:
            break
            
    return best_set
