import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe before 90s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as frozensets
    all_sets = [frozenset(c) for c in combinations(range(1, n+1), k)]
    all_sets.sort()
    
    # For speed, precompute intersections
    # Two sets A, B: A∩B. Three sets A,B,C form sunflower if A∩B == A∩C == B∩C
    # which means A∩B∩C == A∩B (the pairwise intersection equals the triple intersection)
    
    def is_sunflower(a, b, c):
        """Check if three sets form a sunflower (all three pairwise intersections equal)."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    # Greedy with randomized restarts
    best_family = []
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        random.seed(trial)
        shuffled = all_sets[:]
        random.shuffle(shuffled)
        
        family = []
        family_set = set()
        
        for s in shuffled:
            if time.time() > deadline - 5:
                break
            
            compatible = True
            # Check against all pairs in family
            for i in range(len(family)):
                if not compatible:
                    break
                for j in range(i+1, len(family)):
                    if is_sunflower(family[i], family[j], s):
                        compatible = False
                        break
            
            if compatible:
                family.append(s)
                family_set.add(s)
        
        if len(family) > len(best_family):
            best_family = family[:]
            best_family_set = family_set.copy()
        
        # If we've found a very good solution, we can stop
        if len(best_family) >= 200:
            break
    
    # Try local improvement on best found
    if time.time() < deadline - 10 and best_family:
        current = best_family[:]
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding sets not in family
            for s in all_sets:
                if time.time() > deadline - 5:
                    break
                if s in current_set:
                    continue
                
                compatible = True
                for i in range(len(current)):
                    if not compatible:
                        break
                    for j in range(i+1, len(current)):
                        if is_sunflower(current[i], current[j], s):
                            compatible = False
                            break
                
                if compatible:
                    current.append(s)
                    current_set.add(s)
                    improved = True
    
    # Convert to sorted lists
    result = [sorted(s) for s in current if s in set(best_family) or True]
    # Actually let's just use current
    result = [sorted(s) for s in current]
    
    # Verify no duplicates
    seen = set()
    unique = []
    for r in result:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique.append(r)
    
    return unique[:250]
