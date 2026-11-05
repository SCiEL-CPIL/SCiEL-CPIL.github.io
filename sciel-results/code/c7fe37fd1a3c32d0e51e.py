import random
import time

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        xi = subset[i]
        s_diag = 2 * xi
        if s_diag in sums:
            return False
        sums.add(s_diag)
        for j in range(i + 1, n):
            xj = subset[j]
            s_pair = xi + xj
            if s_pair in sums:
                return False
            sums.add(s_pair)
    return True

def max_sidon_size(A):
    max_size = 0
    for mask in range(1 << 12):
        if mask.bit_count() <= max_size:
            continue
        subset = [A[i] for i in range(12) if mask & (1 << i)]
        if is_sidon(subset):
            max_size = mask.bit_count()
            if max_size == 12:
                break
    return max_size

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    current_set = list(range(12))
    current_score = max_sidon_size(current_set)
    
    best_set = current_set[:]
    best_score = current_score
    
    for start in range(0, 128 - 11 + 1):
        if time.time() > deadline:
            break
        s = list(range(start, start + 12))
        score = max_sidon_size(s)
        if score < best_score:
            best_score = score
            best_set = s[:]
            
    while time.time() < deadline:
        i = random.randint(0, 11)
        # Bias towards dense regions
        if random.random() < 0.7:
            new_val = random.randint(0, 40)
        else:
            new_val = random.randint(0, 128)
            
        if new_val in best_set:
            continue
            
        old_val = best_set[i]
        best_set[i] = new_val
        
        new_score = max_sidon_size(best_set)
        
        if new_score < best_score:
            best_score = new_score
            if best_score <= 4:
                # Try a bit more to see if we can get 3
                pass
        else:
            best_set[i] = old_val
            
    return best_set
