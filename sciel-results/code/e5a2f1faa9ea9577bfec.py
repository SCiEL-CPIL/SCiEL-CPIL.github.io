import random
import itertools
import time

def is_non_sidon_4(sorted_quad):
    """
    Returns True if the sorted 4-element tuple is NOT a Sidon set.
    """
    a, b, c, d = sorted_quad
    if a + c == 2 * b: return True
    if a + d == 2 * b: return True
    if b + d == 2 * c: return True
    if a + d == 2 * c: return True
    if a + d == b + c: return True
    return False

def count_non_sidon_4_subsets(S):
    """
    Count the number of 4-element subsets of S that are NOT Sidon.
    """
    S = sorted(S)
    count = 0
    for quad in itertools.combinations(S, 4):
        if is_non_sidon_4(quad):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_count = -1
    
    # Initial candidates
    candidates = [
        list(range(12)),
        list(range(0, 128, 10)),
        [0, 1, 2, 4, 8, 16, 32, 64, 7, 15, 31, 63],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 7, 15, 31, 63, 127, 2, 4, 8, 16]
    ]
    
    for _ in range(10):
        candidates.append(random.sample(range(129), 12))
        
    for c in candidates:
        if len(c) != 12 or len(set(c)) != 12:
            continue
        if min(c) < 0 or max(c) > 128:
            continue
        
        cnt = count_non_sidon_4_subsets(c)
        if cnt > best_count:
            best_count = cnt
            best_set = c[:]
            
        if best_count == 495:
            return best_set

    current = best_set[:]
    current_count = best_count
    no_improvement_count = 0
    
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        new_val = random.randint(0, 128)
        
        if new_val in current:
            continue
            
        current[idx] = new_val
        new_count = count_non_sidon_4_subsets(current)
        
        if new_count > current_count:
            current_count = new_count
            if current_count > best_count:
                best_count = current_count
                best_set = current[:]
            no_improvement_count = 0
            if best_count == 495:
                return best_set
        else:
            current[idx] = old_val
            no_improvement_count += 1
            
            if no_improvement_count > 500:
                # Try a swap
                i = random.randint(0, 11)
                j = random.randint(0, 11)
                if i != j:
                    current[i], current[j] = current[j], current[i]
                    swap_count = count_non_sidon_4_subsets(current)
                    if swap_count > current_count:
                        current_count = swap_count
                        if current_count > best_count:
                            best_count = current_count
                            best_set = current[:]
                        no_improvement_count = 0
                    else:
                        current[i], current[j] = current[j], current[i]
                no_improvement_count = 0

    return best_set
