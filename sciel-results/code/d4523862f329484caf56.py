"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def conflicts(a, b):
        """Return True if (a*b) % (a+b) == 0 (conflict)."""
        return (a * b) % (a + b) == 0
    
    def can_add(num, s):
        """Check if num can be added to set s (no conflicts with any element)."""
        for x in s:
            if x != num and conflicts(x, num):
                return False
        return True
    
    # Start with all odds - guaranteed valid
    odds = list(range(1, 257, 2))
    best_set = set(odds)
    best_size = len(best_set)
    
    # Precompute conflict pairs for evens with odds
    evens = list(range(2, 257, 2))
    
    # For each even, check which odds it conflicts with
    even_conflicts = {}
    for e in evens:
        conf = set()
        for o in odds:
            if conflicts(e, o):
                conf.add(o)
        even_conflicts[e] = conf
    
    # Greedy: try adding evens that don't conflict with current set
    # Multiple random orderings
    best_greedy = set(odds)
    
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        current = set(odds)
        # Random ordering of evens
        shuffled = evens[:]
        random.shuffle(shuffled)
        
        for e in shuffled:
            if time.time() > deadline - 10:
                break
            # Check if e can be added
            ok = True
            for x in current:
                if conflicts(e, x):
                    ok = False
                    break
            if ok:
                current.add(e)
        
        if len(current) > len(best_greedy):
            best_greedy = set(current)
    
    best_set = best_greedy
    best_size = len(best_set)
    
    # Local search: try to improve by swapping
    # Try removing one element and adding two others
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        current = set(best_set)
        
        # Try each pair removal and see if we can add more
        elems = list(current)
        random.shuffle(elems)
        
        for i in range(min(len(elems), 50)):  # Limit work per iteration
            if time.time() > deadline - 5:
                break
            
            for j in range(i+1, min(len(elems), 50)):
                if time.time() > deadline - 5:
                    break
                
                a, b = elems[i], elems[j]
                trial_set = current - {a, b}
                
                # Try to add candidates
                candidates = [x for x in range(1, 257) if x not in trial_set]
                random.shuffle(candidates)
                
                added = 0
                for c in candidates:
                    if time.time() > deadline - 5:
                        break
                    if can_add(c, trial_set):
                        trial_set.add(c)
                        added += 1
                        if added >= 3:  # Need net gain of at least 1
                            break
                
                if len(trial_set) > len(current):
                    best_set = trial_set
                    best_size = len(best_set)
                    improved = True
                    break
            
            if improved:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
