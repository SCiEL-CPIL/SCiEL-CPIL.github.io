import time
import random
import math

def score_set(U):
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1)/log(1) is undefined, but limit is 1?
                   # Actually if U={0}, |U+U|=1, |U-U|=1, ratio 1.
                   # Denom log(1) = 0.
                   # The formula is 1 + log(ratio)/log(2*max+1).
                   # If max=0, denom=0.
                   # Task says "containing 0".
                   # If U={0}, score is technically undefined or limit.
                   # But we want higher scores.
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0 # log(num) <= 0, so score <= 1.
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(num) / denom

def construct():
    start_time = time.time()
    timeout = 110.0
    
    # Initial good seed
    current_U = [0, 1, 2, 4, 8]
    best_U = current_U[:]
    best_score = score_set(current_U)
    
    # Local Search
    # We will try to improve the set by adding, removing, or swapping elements.
    # We restrict the range of elements to keep max_u manageable.
    # If max_u is too large, the denominator grows, hurting the score.
    # So we should probably keep elements relatively small.
    
    max_val_bound = 1000 # Heuristic bound
    
    while time.time() - start_time < timeout:
        # Generate a neighbor
        move_type = random.choice(['add', 'remove', 'swap'])
        new_U = current_U[:]
        
        if move_type == 'add':
            x = random.randint(1, max_val_bound)
            if x not in new_U:
                new_U.append(x)
        elif move_type == 'remove':
            if len(new_U) > 1:
                # Remove a non-zero element
                candidates = [x for x in new_U if x != 0]
                if candidates:
                    x = random.choice(candidates)
                    new_U.remove(x)
        elif move_type == 'swap':
            if len(new_U) > 1:
                candidates = [x for x in new_U if x != 0]
                if candidates:
                    x = random.choice(candidates)
                    y = random.randint(1, max_val_bound)
                    if y not in new_U:
                        new_U.remove(x)
                        new_U.append(y)
        
        # Evaluate
        if new_U != current_U:
            new_score = score_set(new_U)
            if new_score > best_score:
                best_score = new_score
                best_U = new_U[:]
                current_U = new_U[:]
            else:
                # Accept worse moves with some probability (Simulated Annealing style)
                # But simple hill climbing might get stuck.
                # Let's try a small probability of accepting worse moves to escape local optima.
                if random.random() < 0.1:
                    current_U = new_U[:]
        
        # Occasionally reset to a random small set to explore different regions?
        # Or just stick to the local search around the best found.
        
    return sorted(best_U)
