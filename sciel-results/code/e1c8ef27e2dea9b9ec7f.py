import time
import random

def compute_score(U):
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 0
    # Compute U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Compute U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    import math
    if ratio <= 0:
        return 0
    score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with a good seed
    best_set = [0, 1, 3, 9, 27, 81]
    best_score = compute_score(best_set)
    
    # Local search: try adding elements
    random.seed(42)
    max_val = 200
    
    while time.time() < deadline:
        improved = False
        # Try adding a new element
        for _ in range(50):
            candidate = list(best_set)
            new_elem = random.randint(1, max_val)
            if new_elem not in candidate:
                candidate.append(new_elem)
                candidate = sorted(set(candidate))
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_set = candidate
                    improved = True
                    max_val = max(max_val, new_elem + 50)
        # Try removing an element
        for _ in range(20):
            if len(best_set) > 2:
                candidate = list(best_set)
                idx = random.randint(1, len(candidate)-1)  # Keep 0
                candidate.pop(idx)
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_set = candidate
                    improved = True
        # Try swapping
        for _ in range(20):
            if len(best_set) > 2:
                candidate = list(best_set)
                idx = random.randint(1, len(candidate)-1)
                old = candidate[idx]
                new = random.randint(1, max_val)
                if new not in candidate:
                    candidate[idx] = new
                    candidate = sorted(set(candidate))
                    score = compute_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_set = candidate
                        improved = True
        if not improved:
            break
    
    return sorted(set(best_set))
