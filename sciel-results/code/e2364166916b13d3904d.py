import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Greedy Construction - start from largest to maximize score
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    # 2. Local Search with swaps
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        chosen_set = set(chosen)
        
        # Try all possible swaps: replace b in chosen with a not in chosen
        # where a > b and a is coprime to all other elements
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            
            # Check if a is coprime to all in chosen except potentially one
            # Find which elements in chosen are NOT coprime to a
            blockers = []
            for i, b in enumerate(chosen):
                if gcd(a, b) != 1:
                    blockers.append(i)
            
            # If no blockers, a can't be added (set is maximal), so skip
            # If exactly one blocker, we can swap it out if a > b
            if len(blockers) == 0:
                continue
            if len(blockers) > 1:
                continue
            
            i = blockers[0]
            b = chosen[i]
            
            # Only swap if a > b (higher score)
            if a > b:
                # Perform swap
                chosen[i] = a
                chosen_set.remove(b)
                chosen_set.add(a)
                improved = True
        
        # Also try: remove a low-value element and add multiple higher-value ones?
        # For now, stick with single swaps for efficiency
    
    return chosen
