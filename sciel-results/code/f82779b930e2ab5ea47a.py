import time
import math

def _score(U):
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    
    sum_set = set()
    diff_set = set()
    n = len(U)
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sum_set.add(ui + uj)
            diff_set.add(ui - uj)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    U = [0]
    best_score = _score(U)
    
    # Greedy construction
    for _ in range(15):
        if time.time() > deadline - 10.0:
            break
        best_x = -1
        best_local_score = best_score
        current_max = max(U)
        
        # Search for best next element
        # Limit search range to avoid timeout
        # For small N, optimal x is small
        search_limit = 200
        for x in range(current_max + 1, current_max + search_limit + 1):
            if time.time() > deadline - 10.0:
                break
            new_U = U + [x]
            s = _score(new_U)
            if s > best_local_score:
                best_local_score = s
                best_x = x
        
        if best_x == -1:
            break
        U.append(best_x)
        best_score = best_local_score
        
    # Local Search: Try to improve by swapping or adding
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_score = _score(U)
        
        # Try adding
        current_max = max(U)
        for x in range(current_max + 1, current_max + 50):
            if time.time() > deadline - 5.0:
                break
            new_U = U + [x]
            s = _score(new_U)
            if s > current_score:
                U = new_U
                current_score = s
                improved = True
        
        # Try removing
        for i in range(len(U)):
            if time.time() > deadline - 5.0:
                break
            if U[i] == 0:
                continue
            new_U = U[:i] + U[i+1:]
            if not new_U:
                continue
            s = _score(new_U)
            if s > current_score:
                U = new_U
                current_score = s
                improved = True
        
        # Try replacing
        for i in range(len(U)):
            if time.time() > deadline - 5.0:
                break
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try replacing with larger values
            for x in range(old_val + 1, old_val + 50):
                if time.time() > deadline - 5.0:
                    break
                new_U = U[:]
                new_U[i] = x
                s = _score(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    improved = True
                    break

    return U
