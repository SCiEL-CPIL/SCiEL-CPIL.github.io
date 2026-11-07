"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    best_set = None
    best_size = 0
    
    # Try structured constructions first
    for K in range(64, 40, -1):
        H = set(range(K, 129))
        max_L = (K - 1) // 2
        if max_L < 1:
            continue
        start_L = max_L // 2 + 1
        L = set(range(start_L, max_L + 1))
        S = L | H
        if is_valid(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    if best_set is None:
        best_set = set(range(16, 32)) | set(range(64, 129))
        best_size = len(best_set)
    
    # Local search to improve
    current = set(best_set)
    current_size = best_size
    
    # Try adding elements
    for elem in range(1, 129):
        if elem in current:
            continue
        new_set = current | {elem}
        if is_valid(new_set):
            current = new_set
            current_size += 1
    
    if current_size > best_size:
        best_size = current_size
        best_set = set(current)
    
    # Try removing and re-adding different elements
    if time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            # Try random swap: remove one, add another
            removed = None
            added = None
            for _attempt in range(50):
                if time.time() > deadline:
                    break
                if not current:
                    break
                candidates_to_remove = list(current)
                random.shuffle(candidates_to_remove)
                for rem in candidates_to_remove:
                    trial = set(current)
                    trial.discard(rem)
                    # Try to add a new element
                    for add in range(1, 129):
                        if add in trial:
                            continue
                        new_trial = trial | {add}
                        if len(new_trial) > len(trial) and is_valid(new_trial):
                            current = new_trial
                            removed = rem
                            added = add
                            break
                    if added:
                        break
                if added:
                    break
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
    
    # Try greedy addition from scratch with different orderings
    if time.time() < deadline:
        for _restart in range(20):
            if time.time() > deadline:
                break
            trial = set()
            elements = list(range(1, 129))
            random.shuffle(elements)
            for elem in elements:
                new_trial = trial | {elem}
                if is_valid(new_trial):
                    trial = new_trial
            if len(trial) > best_size:
                best_size = len(trial)
                best_set = set(trial)
    
    # Final polish: try to add any remaining elements
    for elem in range(1, 129):
        if elem not in best_set:
            new_set = best_set | {elem}
            if is_valid(new_set):
                best_set = new_set
                best_size += 1
    
    return list(best_set)

def is_valid(S):
    """Check if set S has no violating triples."""
    S_set = set(S)
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S_list[k]
                ac = a + c
                bc = b + c
                if ac in S_set and bc in S_set:
                    return False
    return True

# EVOLVE-BLOCK-END
