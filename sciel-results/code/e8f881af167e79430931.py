import math
import random
import time

def calculate_score(U):
    if not U:
        return 0
    U = sorted(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for u1 in U:
        for u2 in U:
            sum_set.add(u1 + u2)
            diff_set.add(u1 - u2)
    
    ratio = len(diff_set) / len(sum_set)
    # Score formula: 1 + log(ratio) / log(2 * max_u + 1)
    # Note: log is natural log in Python math.log
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Initialize with some good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 4],
        [0, 1, 4, 5],
        [0, 1, 3, 4, 6],
    ]
    
    # Local search / Hill Climbing
    def local_search(initial_set):
        current_set = list(initial_set)
        current_score = calculate_score(current_set)
        
        while time.time() < deadline:
            improved = False
            # Try adding a new element
            max_val = max(current_set)
            # Search range for new elements
            search_max = max_val * 2 + 10
            if len(current_set) < 10:
                search_max = max(search_max, 50)
            
            # Try all integers in a reasonable range
            # To be efficient, we only try a subset or smart range
            # For small sets, we can try all up to search_max
            if search_max <= 200:
                for new_val in range(search_max + 1):
                    if new_val in current_set:
                        continue
                    new_set = current_set + [new_val]
                    new_set.sort()
                    new_score = calculate_score(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break # Restart search with new best
            
            # Try removing an element (if size > 2)
            if len(current_set) > 2:
                for i in range(len(current_set)):
                    if current_set[i] == 0:
                        continue
                    new_set = current_set[:i] + current_set[i+1:]
                    new_score = calculate_score(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
            
            if not improved:
                break
                
        return current_set, current_score

    # Run local search from each candidate
    for cand in candidates:
        if time.time() > deadline:
            break
        s, sc = local_search(cand)
        if sc > best_score:
            best_score = sc
            best_set = s

    return best_set
