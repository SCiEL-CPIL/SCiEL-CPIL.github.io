import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    best_score = float('inf')
    best_c = None
    
    def evaluate(c):
        C = np.convolve(c, c)
        return np.count_nonzero(C)

    def local_search(c, max_iters=200):
        c = np.array(c, dtype=int)
        current_score = evaluate(c)
        for _ in range(max_iters):
            if time.time() > deadline:
                return current_score, c
            improved = False
            for i in range(12):
                old_val = c[i]
                # Generate candidates
                candidates = set()
                candidates.add(old_val + 1)
                candidates.add(old_val - 1)
                candidates.add(1)
                candidates.add(-1)
                candidates.add(2)
                candidates.add(-2)
                candidates.add(random.randint(-20, 20))
                
                best_val = old_val
                best_new_score = current_score
                
                for val in candidates:
                    if val == 0: continue
                    if val == old_val: continue
                    c[i] = val
                    score = evaluate(c)
                    if score < best_new_score:
                        best_new_score = score
                        best_val = val
                c[i] = best_val
                if best_val != old_val:
                    current_score = best_new_score
                    improved = True
            
            if not improved:
                # Random restart of a few variables
                for _ in range(3):
                    i = random.randint(0, 11)
                    new_val = random.randint(-20, 20)
                    if new_val == 0: new_val = 1
                    c[i] = new_val
                current_score = evaluate(c)
                
        return current_score, c

    # Run restarts
    while time.time() < deadline:
        # Init with small random values
        c = np.array([random.randint(-3, 3) for _ in range(12)], dtype=int)
        c[c == 0] = 1
        
        score, c = local_search(c, max_iters=100)
        
        if score < best_score:
            best_score = score
            best_c = c

    if best_c is None:
        best_c = np.ones(12, dtype=int)
        
    return list(best_c)
