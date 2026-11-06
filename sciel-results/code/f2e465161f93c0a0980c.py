import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    N = 32
    RANGE = 512
    
    def evaluate(S):
        diffs = set()
        S_list = list(S)
        # Sort to ensure a < b < c? No, just iterate pairs.
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                d = b - a
                c = b + d
                if c <= RANGE and c in S:
                    diffs.add(d)
        return len(diffs)

    best_score = 0
    best_set = set()
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        
        # Initialization
        if restarts % 10 == 0:
            # Structured init
            S = set()
            # Try to cover small d's
            # 1, 2, 4, 8, 16, 32, 64, 128, 256
            # This covers d=1, 2, 4, 8, 16, 32, 64, 128
            # Add some randoms
            S.update([1, 2, 4, 8, 16, 32, 64, 128, 256])
            while len(S) < N:
                S.add(random.randint(1, RANGE))
        else:
            S = set(random.sample(range(1, RANGE + 1), N))
            
        current_score = evaluate(S)
        if current_score > best_score:
            best_score = current_score
            best_set = set(S)
            
        T = 20.0
        T_min = 0.1
        cooling = 0.9999
        
        steps = 0
        max_steps = 50000 # Increase steps
        while steps < max_steps and time.time() - start_time < time_limit:
            steps += 1
            x = random.choice(list(S))
            y = random.randint(1, RANGE)
            if y in S:
                continue
                
            S.remove(x)
            S.add(y)
            new_score = evaluate(S)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(S)
            else:
                S.remove(y)
                S.add(x)
                
            T *= cooling
            if T < T_min:
                break
                
    return list(best_set)
