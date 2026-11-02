import time
import random
import math

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    if len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    
    # Compute sumset and differenceset
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return -1
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return -1
    
    return 1 + math.log(n_diffs / n_sums) / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = -1
    best_set = [0, 1, 3]
    
    def update_best(cand):
        nonlocal best_score, best_set
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
        return s
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 10],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 6, 12],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 7, 15],
        [0, 2, 5, 12],
        [0, 1, 3, 8],
        [0, 1, 4, 9],
        [0, 2, 5, 11],
        [0, 1, 3, 4, 9],
        [0, 1, 2, 3, 7],
        [0, 1, 3, 5, 11],
        [0, 1, 2, 4, 9],
        [0, 1, 3, 6, 13],
        [0, 1, 4, 8, 17],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 2, 4, 6, 11],
        [0, 1, 3, 5, 9, 18],
        [0, 1, 2, 3, 5, 10],
        [0, 1, 4, 7, 13],
        [0, 1, 2, 5, 9, 18],
        [0, 1, 3, 6, 10, 21],
        [0, 1, 2, 4, 7, 14],
        [0, 1, 3, 5, 8, 17],
        [0, 1, 2, 3, 6, 12],
        [0, 1, 4, 6, 10, 21],
    ]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        update_best(cand)
    
    # Local search
    def local_search(current_set, max_time):
        nonlocal best_score, best_set
        current_set = sorted(set(current_set))
        current_score = score_set(current_set)
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            current_set = sorted(set(current_set))
            current_score = score_set(current_set)
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            
            max_elem = max(current_set) if current_set else 0
            max_search = min(max_elem + 15, 45)
            
            # Try adding
            for new_elem in range(0, max_search + 1):
                if new_elem in current_set:
                    continue
                candidate = sorted(current_set + [new_elem])
                s = score_set(candidate)
                if s > current_score:
                    current_set = candidate
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = list(current_set)
                    break
            
            # Try removing
            for i in range(1, len(current_set)):
                candidate = [x for j, x in enumerate(current_set) if j != i]
                if len(candidate) < 2:
                    continue
                s = score_set(candidate)
                if s > current_score:
                    current_set = candidate
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = list(current_set)
                    break
            
            # Try replacing
            for i in range(1, len(current_set)):
                for new_val in range(0, max_search + 1):
                    if new_val == current_set[i]:
                        continue
                    if new_val in current_set:
                        continue
                    candidate = list(current_set)
                    candidate[i] = new_val
                    candidate = sorted(set(candidate))
                    if 0 not in candidate or len(candidate) < 2:
                        continue
                    s = score_set(candidate)
                    if s > current_score:
                        current_set = candidate
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = list(current_set)
                        break
                if improved:
                    break
        
        return current_set
    
    # Local search from best
    if time.time() < deadline:
        local_search(best_set, deadline)
    
    # Random restarts
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline:
            break
        size = random.randint(4, 10)
        elems = [0]
        for _ in range(size - 1):
            elems.append(random.randint(1, 35))
        elems = sorted(set(elems))
        if len(elems) < 2:
            continue
        update_best(elems)
        if time.time() < deadline:
            local_search(elems, deadline)
    
    return best_set
