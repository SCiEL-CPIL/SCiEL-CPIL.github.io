import random
import math
import time

def score_set(U):
    if not U or 0 not in U:
        return -1
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    p = len(sum_set)
    d = len(diff_set)
    if p == 0:
        return -1
    ratio = d / p
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 6],
        [0, 1, 2, 4],
        [0, 1, 4, 6],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 12],
        [0, 1, 3, 7, 10],
        [0, 1, 4, 6, 12],
        [0, 1, 5, 7, 14],
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    # Local search
    current_set = best_set[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Randomly modify the set
        new_set = current_set[:]
        action = random.choice(['add', 'remove', 'replace'])
        
        if action == 'add' and len(new_set) < 10:
            # Add a random number
            max_val = max(new_set) if new_set else 0
            new_val = random.randint(1, max_val * 2 + 10)
            if new_val not in new_set:
                new_set.append(new_val)
        elif action == 'remove' and len(new_set) > 2:
            # Remove a random non-zero element
            idx = random.randint(1, len(new_set) - 1)
            new_set.pop(idx)
        elif action == 'replace' and len(new_set) > 2:
            # Replace a random non-zero element
            idx = random.randint(1, len(new_set) - 1)
            max_val = max(new_set) if new_set else 0
            new_val = random.randint(1, max_val * 2 + 10)
            new_set[idx] = new_val
            
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            new_set.append(0)
            
        new_score = score_set(new_set)
        
        if new_score > current_score:
            current_score = new_score
            current_set = new_set
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            # Occasionally accept worse moves to escape local maxima
            if random.random() < 0.1:
                current_set = new_set
                current_score = new_score

    return best_set
