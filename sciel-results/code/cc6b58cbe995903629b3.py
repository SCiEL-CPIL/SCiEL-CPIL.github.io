import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    def greedy_with_lookahead(rng):
        S = []
        reachable = 1  # bit 0 set
        candidates = list(range(1, 101))
        
        while candidates:
            best_c = None
            best_score = -1
            best_R_new = 0
            
            # Find the best candidate to add
            # We iterate through all candidates to find the one that leaves the most options open
            for c in candidates:
                # Check if c is valid
                target = 200 - c
                if target >= 0 and ((reachable >> target) & 1):
                    continue
                
                # Simulate adding c
                R_new = reachable | (reachable << c)
                R_new &= ((1 << 201) - 1)
                
                # Score: how many other candidates remain valid?
                score = 0
                # Optimization: if score is already lower than best_score, we could prune?
                # But we need to check all d to get exact score.
                # However, if best_score is high, we might not need exact count for low potential c.
                # But N is small, so just count all.
                for d in candidates:
                    if d == c:
                        continue
                    target_d = 200 - d
                    if target_d < 0:
                        score += 1
                        continue
                    if not ((R_new >> target_d) & 1):
                        score += 1
                
                # Tie-breaking: prefer smaller c? Or random?
                # Using random noise helps escape local minima in the greedy path.
                noise = rng.random() * 0.1
                total_score = score + noise
                
                if total_score > best_score:
                    best_score = total_score
                    best_c = c
                    best_R_new = R_new
            
            if best_c is None:
                break
            
            # Add best_c
            S.append(best_c)
            reachable = best_R_new
            candidates.remove(best_c)
            
        return S

    rng = random.Random(42)
    
    # Run multiple trials
    # Each trial is fast.
    while time.time() < deadline:
        s = greedy_with_lookahead(rng)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return best_set
