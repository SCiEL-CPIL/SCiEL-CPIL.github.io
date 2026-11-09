import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0
    
    def is_valid_addition(S_list, y):
        # Check 1: y divides sum of 3 in S_list
        # Precompute sums of triples? 
        # For small N, just iterate.
        for a, b, c in combinations(S_list, 3):
            if (a + b + c) % y == 0:
                return False
        
        # Check 2: x in S_list divides sum of 3 in S_list U {y} including y
        # i.e., x divides y + u + v for u, v in S_list \ {x}
        # Precompute pairs
        pairs = list(combinations(S_list, 2))
        for x in S_list:
            if x == y:
                continue
            # Filter pairs not involving x
            # Optimization: If x is large, few pairs might conflict? No.
            for u, v in pairs:
                if u == x or v == x:
                    continue
                if (y + u + v) % x == 0:
                    return False
        return True

    best_size = 0
    best_set = []
    candidates = list(range(1, 193))
    
    while time.time() < deadline:
        # Randomize order for diversity
        current_candidates = candidates[:]
        random.shuffle(current_candidates)
        
        # Optional: Bias towards larger numbers in some restarts
        # 50% chance descending, 50% random
        if random.random() < 0.5:
            current_candidates.sort(reverse=True)
            
        S = []
        S_set = set()
        
        # Greedy
        for x in current_candidates:
            if x in S_set:
                continue
            if is_valid_addition(S, x):
                S.append(x)
                S_set.add(x)
                
        # Local Search
        stagnation = 0
        max_stagnation = 20 # Reduced to allow more restarts
        
        while time.time() < deadline and stagnation < max_stagnation:
            added = False
            swapped = False
            
            # Try to add
            for y in current_candidates:
                if y in S_set:
                    continue
                if is_valid_addition(S, y):
                    S.append(y)
                    S_set.add(y)
                    added = True
                    stagnation = 0
                    break
            
            if not added:
                if S:
                    idx = random.randint(0, len(S) - 1)
                    x = S[idx]
                    S.pop(idx)
                    S_set.remove(x)
                    
                    for y in current_candidates:
                        if y in S_set:
                            continue
                        if is_valid_addition(S, y):
                            S.append(y)
                            S_set.add(y)
                            swapped = True
                            stagnation = 0
                            break
                    
                    if not swapped:
                        S.append(x)
                        S_set.add(x)
                        stagnation += 1
                else:
                    stagnation += 1

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
