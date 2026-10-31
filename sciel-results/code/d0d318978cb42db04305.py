import math
import time

def calculate_score(U):
    if not U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    
    sumset = set()
    diffset = set()
    n = len(U_set)
    U_list = list(U_set)
    
    for i in range(n):
        for j in range(i, n):
            sumset.add(U_list[i] + U_list[j])
        for j in range(n):
            diffset.add(U_list[i] - U_list[j])
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1 + math.log(ratio) / denom
    return score

def is_golomb(U):
    """Check if U is a Golomb ruler (distinct differences)."""
    U_list = sorted(U)
    n = len(U_list)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = U_list[j] - U_list[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Known optimal Golomb rulers for small n
    # n: [0, ...]
    optimal_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 34, 44], # Approximate, might not be optimal
        10: [0, 1, 4, 10, 18, 23, 29, 34, 44, 54] # Approximate
    }
    
    best_score = -1
    best_set = [0]
    
    # Evaluate known optimal rulers
    for n, U in optimal_rulers.items():
        if time.time() > deadline:
            break
        if is_golomb(U):
            score = calculate_score(U)
            if score > best_score:
                best_score = score
                best_set = U
        else:
            # If hardcoded is not valid (shouldn't happen), try to fix or skip
            pass

    # Try greedy generation for n up to 15 to see if we find better
    # This is a fallback/improvement strategy
    for n in range(1, 16):
        if time.time() > deadline:
            break
            
        U = [0]
        current_max = 0
        for _ in range(n - 1):
            candidate = current_max + 1
            found = False
            while candidate < 1000: # Limit search space
                new_U = U + [candidate]
                if is_golomb(new_U):
                    U = new_U
                    current_max = candidate
                    found = True
                    break
                candidate += 1
            if not found:
                break
                
        if len(U) == n:
            score = calculate_score(U)
            if score > best_score:
                best_score = score
                best_set = U

    return best_set
