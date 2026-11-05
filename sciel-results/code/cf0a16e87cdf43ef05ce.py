import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    n = 255
    target = 255
    
    # All candidates: 1 to 255, must have gcd > 1 with target
    # Actually, we need gcd(a,b) > 1 for ALL pairs, so every element must share a factor with every other
    # Start with just {255}
    
    # Precompute gcd for all pairs involving numbers that could be in the set
    # A number can only be in the set if gcd(num, 255) > 1
    # 255 = 3*5*17, so num must be divisible by 3, 5, or 17
    
    candidates = [x for x in range(1, n + 1) if math.gcd(x, target) > 1]
    
    # Greedy: start with target, add candidates one by one
    # At each step, add the candidate that has gcd>1 with all current set members
    # and maximizes some heuristic
    
    current_set = {target}
    
    # For efficiency, track which candidates are still "compatible" with current set
    # A candidate is compatible if gcd(candidate, x) > 1 for all x in current_set
    
    # Precompute gcd matrix for candidates
    # This is O(|candidates|^2) which is manageable for ~120 candidates
    
    # Let's use a different greedy: repeatedly pick the candidate with the most 
    # "future potential" - i.e., that is compatible with the most remaining candidates
    
    remaining = set(candidates) - {target}
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_candidate = None
        best_score = -1
        
        for cand in remaining:
            # Check if cand is compatible with all in current_set
            compatible = True
            for x in current_set:
                if math.gcd(cand, x) == 1:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Score: number of remaining candidates that are also compatible with cand
            # and with current_set (approximately)
            score = 0
            for other in remaining:
                if other == cand:
                    continue
                # Quick check: does other share a factor with cand?
                if math.gcd(cand, other) > 1:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_candidate = cand
        
        if best_candidate is None:
            break
        
        current_set.add(best_candidate)
        remaining.remove(best_candidate)
    
    return list(current_set)
