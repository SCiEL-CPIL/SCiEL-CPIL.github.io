import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_score = 0
    best_set = None
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Initialize: random set of 32 elements from 1..512
        S = set(random.sample(range(1, 513), 32))
        S_bool = np.zeros(513, dtype=bool)
        for x in S:
            S_bool[x] = True
        
        cur_score = _score_set(S_bool)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)
        
        # Local search with simulated annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
            
            # Pick random element to remove and random element to add
            x = random.choice(list(S))
            y = random.choice([i for i in range(1, 513) if i not in S])
            
            # Try swap
            S_bool[x] = False
            S_bool[y] = True
            new_score = _score_set(S_bool)
            
            delta = new_score - cur_score
            if delta > 0 or random.random() < np.exp(delta / T):
                # Accept
                S.discard(x)
                S.add(y)
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                S_bool[x] = True
                S_bool[y] = False
                no_improve += 1
            
            T = max(T_min, T * cooling)
        
        # Reset temperature for next restart
        T = 5.0
    
    if best_set is None:
        return list(range(1, 33))
    
    return list(best_set)


def _score_set(S_bool):
    covered = 0
    for d in range(1, 256):
        max_a = 512 - 2*d
        if max_a < 1:
            break
        arr1 = S_bool[1:1+max_a]
        arr2 = S_bool[1+d:1+d+max_a]
        arr3 = S_bool[1+2*d:1+2*d+max_a]
        if np.any(arr1 & arr2 & arr3):
            covered += 1
    return covered
