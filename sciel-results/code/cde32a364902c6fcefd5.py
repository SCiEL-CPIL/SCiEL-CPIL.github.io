import time
import random
import math

def score_set(U):
    if not U:
        return 0.0
    if 0 not in U:
        U = U | {0}
    U = set(U)
    sum_set = {a + b for a in U for b in U}
    diff_set = {a - b for a in U for b in U}
    max_u = max(U)
    if max_u == 0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_U = [0, 1, 4, 6]
    
    # Verify initial best
    s = score_set(best_U)
    if s > best_score:
        best_score = s
        
    current_U = best_U[:]
    
    while time.time() < deadline:
        r = random.random()
        if r < 0.4:
            # Add element
            max_val = max(current_U)
            new_val = random.randint(0, max_val + 5)
            if new_val not in current_U:
                neighbor = current_U + [new_val]
            else:
                continue
        elif r < 0.7:
            # Remove element (not 0)
            if len(current_U) > 1:
                removable = [x for x in current_U if x != 0]
                if removable:
                    rem_val = random.choice(removable)
                    neighbor = [x for x in current_U if x != rem_val]
                else:
                    continue
            else:
                continue
        else:
            # Move element
            if len(current_U) > 1:
                idx = random.randint(0, len(current_U) - 1)
                if current_U[idx] == 0:
                    # Find non-zero index
                    non_zero = [i for i, x in enumerate(current_U) if x != 0]
                    if not non_zero:
                        continue
                    idx = random.choice(non_zero)
                old_val = current_U[idx]
                new_val = random.randint(0, max(current_U) + 5)
                if new_val in current_U and new_val != old_val:
                    continue
                neighbor = current_U[:]
                neighbor[idx] = new_val
            else:
                continue
                
        s = score_set(neighbor)
        if s > best_score:
            best_score = s
            best_U = neighbor[:]
            current_U = neighbor[:]
        elif s > score_set(current_U) * 0.95:
            # Simulated annealing-like acceptance
            if random.random() < 0.1:
                current_U = neighbor[:]
                
    return sorted(best_U)
