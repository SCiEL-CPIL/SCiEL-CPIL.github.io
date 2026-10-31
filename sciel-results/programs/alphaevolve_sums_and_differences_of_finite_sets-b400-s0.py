import math
import random
import time

def score(U):
    if not U:
        return 0
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 3, 7]
    best_score = score(best_set)
    
    # Simple hill climbing
    current_set = list(best_set)
    current_score = best_score
    
    while time.time() < deadline:
        # Generate neighbor
        neighbor = list(current_set)
        move = random.choice(['add', 'remove', 'shift'])
        
        if move == 'add':
            max_val = max(neighbor)
            # Try adding small numbers or numbers near max
            candidates = [i for i in range(1, max_val + 5) if i not in neighbor]
            if candidates:
                neighbor.append(random.choice(candidates))
        elif move == 'remove':
            if len(neighbor) > 1:
                # Don't remove 0
                rem = random.choice([x for x in neighbor if x != 0])
                neighbor.remove(rem)
        else:
            # Shift a random element
            if len(neighbor) > 1:
                idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                old_val = neighbor[idx]
                new_val = random.randint(1, max(neighbor) + 5)
                if new_val not in neighbor:
                    neighbor[idx] = new_val
        
        neighbor.sort()
        new_score = score(neighbor)
        
        if new_score > current_score:
            current_set = neighbor
            current_score = new_score
            if current_score > best_score:
                best_set = current_set
                best_score = current_score
        else:
            # With some probability, accept worse to escape local optima
            if random.random() < 0.1:
                current_set = neighbor
                current_score = new_score

    return best_set
