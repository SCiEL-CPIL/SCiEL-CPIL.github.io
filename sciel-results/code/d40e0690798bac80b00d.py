import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def get_score(S):
        S_set = set(S)
        diffs = set()
        # Iterate pairs to find valid differences d
        # x, y in S, x < y. d = y - x.
        # Check if x + 2d = 2y - x is in S.
        for x in S_set:
            for y in S_set:
                if x >= y:
                    continue
                d = y - x
                z = 2 * y - x
                if z in S_set:
                    diffs.add(d)
        return len(diffs)

    def random_set():
        return random.sample(range(1, 513), 32)
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    candidates.append(list(range(1, 17)) + list(range(100, 116)))
    candidates.append(list(range(1, 9)) + list(range(50, 58)) + list(range(100, 108)) + list(range(150, 158)))
    for _ in range(10):
        candidates.append(random_set())
        
    best_score = -1
    best_set = None
    
    for cand in candidates:
        s = get_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # Simulated Annealing
    current_set = list(best_set)
    current_score = best_score
    T = 1.0
    cooling = 0.9995  # Slow cooling
    
    while time.time() < deadline:
        # Move: Replace one element with a random new one
        idx = random.randint(0, 31)
        old_elem = current_set[idx]
        new_elem = random.randint(1, 512)
        
        # Ensure distinct
        if new_elem in current_set:
            continue
            
        # Create new set
        new_set = list(current_set)
        new_set[idx] = new_elem
        
        new_score = get_score(new_set)
        
        # Acceptance
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            delta = new_score - current_score
            if delta < 0:
                # Avoid division by zero or underflow issues
                if T > 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_set = new_set
                        current_score = new_score
        
        T *= cooling
        if T < 0.001:
            # Restart from best found so far
            current_set = list(best_set)
            current_score = best_score
            T = 1.0

    # Ensure the returned set is valid
    # It should be, but just in case
    if len(set(best_set)) != 32 or not all(1 <= x <= 512 for x in best_set):
        return list(range(1, 33))
        
    return best_set
