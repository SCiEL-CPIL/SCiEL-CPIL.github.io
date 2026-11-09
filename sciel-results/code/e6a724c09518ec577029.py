import random
import time
import math

def _check_set(S):
    n = len(S)
    if n < 2:
        return [True] * n
    
    valid = [True] * n
    max_sum = sum(S)
    
    for i in range(n):
        x = S[i]
        S_minus_x = [S[j] for j in range(n) if j != i]
        
        # bits[k] is an integer where bit s is 1 if sum s is achievable by a subset of size k
        bits = [0] * n
        bits[0] = 1
        current_max_k = 0
        
        for v in S_minus_x:
            # Update in reverse order
            # Max k we can reach is current_max_k + 1
            upper = min(current_max_k + 1, n - 1)
            for k in range(upper, 0, -1):
                bits[k] |= (bits[k - 1] << v)
            current_max_k += 1
            
        # Check mean condition
        # k must be >= 2
        # Also k <= len(S_minus_x)
        for k in range(2, min(n, current_max_k + 1)):
            target = k * x
            if target > max_sum:
                break
            if (bits[k] >> target) & 1:
                valid[i] = False
                break
                
    return valid

def construct():
    start_time = time.time()
    limit = 512
    max_size = 48
    
    # Greedy initialization: try to build a valid set greedily
    def greedy_init():
        S = []
        candidates = list(range(1, limit + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= max_size:
                break
            # Check if x can be added
            # We need to check if x is mean of subset of S
            # And if any element in S becomes mean of subset of S U {x}
            
            # This is expensive to do fully.
            # For init, let's just pick and let SA fix it.
            S.append(x)
            
        return S

    S = greedy_init()
    current_valid = _check_set(S)
    current_score = sum(current_valid)
    best_S = S[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < 55:
        # Pick a random index to mutate
        idx = random.randint(0, max_size - 1)
        old_val = S[idx]
        
        # Pick a new value not in S
        new_val = random.randint(1, limit)
        attempts = 0
        while new_val in S and attempts < 50:
            new_val = random.randint(1, limit)
            attempts += 1
        if new_val in S:
            continue
            
        S[idx] = new_val
        
        new_valid = _check_set(S)
        new_score = sum(new_valid)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            current_valid = new_valid
            if current_score > best_score:
                best_score = current_score
                best_S = S[:]
                if best_score == max_size:
                    break
        else:
            S[idx] = old_val
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_S
