"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    
    # Start with the known valid solution: all numbers > n/2
    # This gives n//2 elements
    base_set = set(range(n//2 + 1, n + 1))
    best = list(base_set)
    best_size = len(best)
    
    # Try to add small numbers one by one, checking validity
    # A set S is valid if for no three distinct a,b,c in S are a+b, a+c, b+c all in S
    
    def is_valid(s):
        """Check if set s is valid (no forbidden triple)."""
        s_list = list(s)
        s_set = set(s)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                ab = a + b
                if ab not in s_set:
                    continue
                for k in range(j+1, len(s_list)):
                    c = s_list[k]
                    ac = a + c
                    bc = b + c
                    if ac in s_set and bc in s_set:
                        return False
        return True
    
    # Greedy: try adding each small number, keep if valid
    current = set(base_set)
    for x in range(1, n//2 + 1):
        if time.time() > deadline:
            break
        trial = current | {x}
        if is_valid(trial):
            current = trial
    
    if len(current) > best_size:
        best = list(current)
        best_size = len(current)
    
    # Local search: try swapping or adding/removing
    # Try to add more numbers
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            trial = current | {x}
            if is_valid(trial):
                current = trial
                improved = True
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
        if not improved:
            break
    
    # Try removing some large numbers and adding multiple small ones
    # This is a more complex search - try random perturbations
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        trial = set(current)
        # Remove 1-3 random elements
        if len(trial) > 0:
            to_remove = random.sample(list(trial), min(3, len(trial)))
            for x in to_remove:
                trial.discard(x)
        
        # Try to add back different elements
        for x in range(1, n + 1):
            if x in trial:
                continue
            t2 = trial | {x}
            if is_valid(t2):
                trial = t2
        
        if len(trial) > len(current):
            current = trial
            if len(current) > best_size:
                best = list(current)
                best_size = len(current)
    
    # Final check
    if is_valid(set(best)):
        return sorted(best)
    else:
        return sorted(list(base_set))
# EVOLVE-BLOCK-END
