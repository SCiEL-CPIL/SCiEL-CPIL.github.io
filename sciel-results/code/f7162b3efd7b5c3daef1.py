import numpy as np

def find_max_sidon_set(n):
    best_set = []
    
    def backtrack(start, current_set, used_sums):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = list(current_set)
        
        for i in range(start, n):
            # Check if i can be added
            valid = True
            new_sums = []
            for j in current_set:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if valid:
                # Check i+i
                s_ii = i + i
                if s_ii in used_sums:
                    valid = False
                else:
                    new_sums.append(s_ii)
            
            if valid:
                # Add i
                current_set.append(i)
                used_sums.update(new_sums)
                backtrack(i + 1, current_set, used_sums)
                # Undo
                current_set.pop()
                for s in new_sums:
                    used_sums.discard(s)
                    
    backtrack(0, [], set())
    return best_set

def construct():
    # Find max Sidon set in 0..49
    sidon_set = find_max_sidon_set(50)
    h = [0.0] * 50
    k = len(sidon_set)
    if k == 0:
        return [1.0] + [0.0]*49
    val = 1.0 / k
    for idx in sidon_set:
        h[idx] = val
    return h
