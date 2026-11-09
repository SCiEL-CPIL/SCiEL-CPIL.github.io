import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    N = 32
    MAX_VAL = 512
    
    def score_from_list(S_list):
        # S_list is a list of N integers
        S_set = set(S_list)
        diffs = set()
        S_sorted = sorted(S_list)
        for i in range(N):
            for j in range(i+1, N):
                a = S_sorted[i]
                c = S_sorted[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if b in S_set:
                        d = c - a
                        if d > 0:
                            diffs.add(d)
        return len(diffs)
    
    def generate_initial_candidates():
        candidates = []
        
        # 1. Base 2 (consecutive)
        candidates.append(list(range(1, N+1)))
        
        # 2. Base 3 Cantor-like
        S_cantor = []
        for mask in range(1 << 5): # 5 bits -> 32 elements
            val = 0
            for i in range(5):
                if mask & (1 << i):
                    val += 3**i
            S_cantor.append(val + 1)
        candidates.append(S_cantor)
        
        # 3. Base 4 (digits 0, 2)
        S_base4 = []
        for mask in range(1 << 5):
            val = 0
            for i in range(5):
                if mask & (1 << i):
                    val += 2 * (4**i)
            S_base4.append(val + 1)
        candidates.append(S_base4)
        
        # 4. Random
        for _ in range(10):
            S_rand = random.sample(range(1, MAX_VAL+1), N)
            candidates.append(S_rand)
            
        # 5. Two blocks
        S_two = list(range(1, 17)) + list(range(200, 216))
        candidates.append(S_two)
        
        # 6. Three blocks
        S_three = list(range(1, 11)) + list(range(100, 110)) + list(range(300, 310))
        candidates.append(S_three)
        
        return candidates

    candidates = generate_initial_candidates()
    
    best_score = 0
    best_list = candidates[0]
    
    # Simulated Annealing
    T0 = 100.0
    alpha = 0.999
    
    for S_init in candidates:
        if time.time() > deadline - 10.0:
            break
            
        current_list = list(S_init)
        current_set = set(current_list)
        current_score = score_from_list(current_list)
        
        best_local_list = list(current_list)
        best_local_set = set(current_list)
        best_local_score = current_score
        
        T = T0
        steps = 0
        max_steps = 500000
        
        while T > 0.01 and steps < max_steps:
            if time.time() > deadline - 2.0:
                break
                
            # Pick random index for x
            idx_x = random.randint(0, N-1)
            x = current_list[idx_x]
            
            # Pick random y
            y = random.randint(1, MAX_VAL)
            
            if x == y:
                continue
                
            # Swap
            current_list[idx_x] = y
            current_set.remove(x)
            current_set.add(y)
            
            new_score = score_from_list(current_list)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_list = list(current_list)
                    best_local_set = set(current_list)
            else:
                # Revert
                current_list[idx_x] = x
                current_set.remove(y)
                current_set.add(x)
                
            T *= alpha
            steps += 1
            
        # Final Polish: Hill Climbing
        # Use best_local_list
        S_list = list(best_local_list)
        S_set = set(S_list)
        S_score = best_local_score
        
        improved = True
        polish_passes = 0
        while improved and polish_passes < 5:
            improved = False
            polish_passes += 1
            # Iterate over all x in S
            # To avoid index errors during modification, iterate over a copy of the list
            for x in list(S_list):
                if time.time() > deadline - 1.0:
                    break
                # Try to swap x with y
                for y in range(1, MAX_VAL + 1):
                    if y in S_set:
                        continue
                    # Swap x with y
                    # Find index of x
                    # Since S_list is a list, we can use .index(x) which is O(N)
                    # Or maintain a dict mapping value to index?
                    # N is small, .index is fine.
                    idx = S_list.index(x)
                    S_list[idx] = y
                    S_set.remove(x)
                    S_set.add(y)
                    
                    new_s = score_from_list(S_list)
                    
                    if new_s > S_score:
                        S_score = new_s
                        improved = True
                        # Keep the change
                        break
                    else:
                        # Revert
                        S_list[idx] = x
                        S_set.remove(y)
                        S_set.add(x)
                if time.time() > deadline - 1.0:
                    break
            if time.time() > deadline - 1.0:
                break
                
        if S_score > best_score:
            best_score = S_score
            best_list = S_list
            
    return best_list
