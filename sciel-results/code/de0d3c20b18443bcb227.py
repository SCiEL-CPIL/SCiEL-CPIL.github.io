import time
import random

def construct():
    N = 192
    MAX_SIZE = 48
    deadline = time.time() + 85.0
    
    def can_add(S, x):
        K = len(S)
        
        # 1. x divides sum of 3 in S
        for i in range(K):
            a = S[i]
            for j in range(i+1, K):
                b = S[j]
                for k in range(j+1, K):
                    c = S[k]
                    if (a + b + c) % x == 0:
                        return False
        
        # 2. y in S divides sum of x and 2 in S\{y}
        for i in range(K):
            y = S[i]
            for j in range(K):
                if j == i: continue
                u = S[j]
                for k in range(j+1, K):
                    if k == i: continue
                    v = S[k]
                    if (x + u + v) % y == 0:
                        return False
        
        return True

    def greedy_construct(randomize_order=False, seed=None):
        if seed is not None:
            random.seed(seed)
        S = []
        candidates = list(range(1, N + 1))
        if randomize_order:
            random.shuffle(candidates)
        else:
            candidates.sort(reverse=True)
            
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            if can_add(S, x):
                S.append(x)
        return S

    best_set = []
    
    # Run a few initial greedy searches
    best_set = greedy_construct(randomize_order=False)
    
    for i in range(10):
        s = greedy_construct(randomize_order=True, seed=i)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) == MAX_SIZE:
                return best_set

    # Local Search on best_set
    current_set = list(best_set)
    no_progress_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # Try to add
        candidates = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(current_set, x):
                current_set.append(x)
                improved = True
                if len(current_set) > len(best_set):
                    best_set = list(current_set)
                if len(best_set) == MAX_SIZE:
                    return best_set
                break
        
        if not improved:
            # Try swaps
            swaps_made = False
            for _ in range(20): # Try a few swaps
                if not current_set:
                    break
                i = random.randint(0, len(current_set) - 1)
                y = current_set[i]
                
                candidates = [x for x in range(1, N + 1) if x not in current_set]
                if not candidates:
                    break
                random.shuffle(candidates)
                
                for x in candidates:
                    temp_set = current_set[:i] + current_set[i+1:]
                    if can_add(temp_set, x):
                        current_set = temp_set + [x]
                        swaps_made = True
                        break
                if swaps_made:
                    break
            
            if not swaps_made:
                no_progress_count += 1
                if no_progress_count > 5:
                    # Restart with new greedy
                    new_seed = int(time.time() * 1000) % 10000
                    s = greedy_construct(randomize_order=True, seed=new_seed)
                    if len(s) > len(current_set):
                        current_set = s
                        if len(current_set) > len(best_set):
                            best_set = list(current_set)
                    no_progress_count = 0
            else:
                no_progress_count = 0
        else:
            no_progress_count = 0

    return best_set
