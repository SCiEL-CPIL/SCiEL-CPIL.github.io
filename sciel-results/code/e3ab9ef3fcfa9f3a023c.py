import numpy as np
import random
import time

def construct():
    best_score = 23
    best_c = [1] * 12
    start_time = time.time()
    
    def get_score(c):
        # c is numpy array
        conv = np.convolve(c, c)
        return np.count_nonzero(conv)
    
    for restart in range(100): # Max 100 restarts
        # Check time
        if time.time() - start_time > 55:
            break
            
        # Initialize
        # Use random small integers
        c = np.array([random.randint(-5, 5) for _ in range(12)], dtype=int)
        c[c == 0] = 1
        
        current_score = get_score(c)
        
        # Local search
        steps = 0
        max_steps = 1000
        while steps < max_steps and time.time() - start_time < 55:
            steps += 1
            idx = random.randint(0, 11)
            old_val = c[idx]
            
            # Try a few candidate values
            # Neighbors and random
            candidates = [old_val + random.choice([-2, -1, 1, 2])]
            if random.random() < 0.1:
                candidates.append(random.randint(-20, 20))
                
            improved = False
            for new_val in candidates:
                if new_val == 0: continue
                if abs(new_val) > 64: continue
                if new_val == old_val: continue
                
                c[idx] = new_val
                new_score = get_score(c)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    c[idx] = old_val
            
            if not improved:
                # If no improvement, try a bigger jump or break
                if random.random() < 0.05:
                    c[idx] = random.randint(-20, 20)
                    if c[idx] == 0: c[idx] = 1
                    new_score = get_score(c)
                    if new_score < current_score:
                        current_score = new_score
                    else:
                        c[idx] = old_val
                else:
                    break
                    
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            
        # If we found 3 terms, it's likely optimal, so we can stop early?
        # Min possible is 3.
        if best_score <= 3:
            break

    return [int(x) for x in best_c]
