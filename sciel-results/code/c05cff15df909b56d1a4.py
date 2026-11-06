import time
import random

def score_set(S):
    S = set(S)
    diffs = set()
    # Iterate over all pairs to find potential differences
    # a, b in S, a < b. d = b - a.
    # Check if a + 2d is in S.
    # Note: a + 2d = a + 2(b-a) = 2b - a.
    # So we check if 2b - a is in S.
    for a in S:
        for b in S:
            if a == b: continue
            # We only care about d > 0.
            # If a < b, d = b - a > 0.
            # The progression is a, a+d, a+2d.
            # So we need a, b, 2b-a in S.
            # This covers the case where a is the start.
            # What if b is the start? Then we'd have b, b+d, b+2d.
            # This is covered when we pick the pair (b, b+d) and check b+2d.
            # So iterating all ordered pairs (a,b) with a!=b and checking if the third term exists is sufficient.
            # Actually, just checking if 2*b - a is in S for all a!=b in S is enough to find ALL 3-APs.
            # If a, a+d, a+2d are in S, then for pair (a, a+d), 2(a+d)-a = a+2d is in S.
            # And d = (a+d) - a.
            c = 2 * b - a
            if c in S:
                d = abs(b - a)
                if d > 0:
                    diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0
    best_set = set()
    
    # Initial random sets or structured starts
    # Start with a few diverse initializations
    initial_sets = []
    
    # 1. Consecutive integers
    initial_sets.append(list(range(1, 33)))
    
    # 2. Random sets
    for _ in range(5):
        s = random.sample(range(1, 513), 32)
        initial_sets.append(s)
    
    # 3. Structured: Powers of 2? No, too sparse.
    # Maybe arithmetic progressions with different steps?
    # e.g., 1, 1+k, 1+2k ... but we need 32 elements.
    
    for init in initial_sets:
        current_set = set(init)
        current_score = score_set(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set
            
        # Local Search
        no_improvement_count = 0
        max_no_improvement = 2000
        
        while time.time() < deadline:
            if no_improvement_count > max_no_improvement:
                break
                
            # Pick a random element to remove
            x = random.choice(list(current_set))
            
            # Pick a random element to add from 1..512 not in current_set
            # To make it faster, we can just pick a random y in 1..512 and if it's in set, retry or skip.
            # Better: maintain a list of available numbers? 
            # 512 is small. 
            y = random.randint(1, 512)
            if y in current_set:
                continue
                
            new_set = current_set.copy()
            new_set.remove(x)
            new_set.add(y)
            
            new_score = score_set(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                no_improvement_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set
            else:
                no_improvement_count += 1
                
            # Occasional restart if stuck
            if no_improvement_count > max_no_improvement * 2:
                break

    return list(best_set)
