import time
import random
import math

def construct():
    start_time = time.monotonic()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def get_score(S):
        # S is a list of N distinct ints
        in_set = [False] * (MAX_VAL + 1)
        for x in S:
            in_set[x] = True
        
        valid_d = 0
        # d can be up to (MAX_VAL - 1) // 2
        max_d = MAX_VAL // 2
        for d in range(1, max_d + 1):
            found = False
            # We need a, a+d, a+2d in S
            # a must be <= MAX_VAL - 2d
            # Iterate a in S? Or iterate a in 1..MAX_VAL-2d?
            # Iterating a in S is faster if S is small (32)
            for a in S:
                if a + 2 * d <= MAX_VAL:
                    if in_set[a + d] and in_set[a + 2 * d]:
                        valid_d += 1
                        found = True
                        break
        return valid_d

    # Initial candidates
    candidates = []
    
    # 1. Interval
    candidates.append(list(range(1, N + 1)))
    
    # 2. Two blocks
    b1 = list(range(1, N // 2 + 1))
    b2 = [x + 250 for x in range(N // 2, N)]
    candidates.append(b1 + b2)
    
    # 3. Random
    random.seed(42)
    candidates.append(random.sample(range(1, MAX_VAL + 1), N))
    
    # 4. "Stretched" interval: 1, 1+K, 1+2K...
    # Try to cover more d's
    # If we take 1, 1+10, 1+20... we get d=10, 20, 30...
    # But we lose small d's.
    # Maybe a mix?
    
    best_score = -1
    best_set = None
    
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    current_set = best_set[:]
    current_score = best_score
    
    T = 50.0
    T_min = 0.5
    cooling_rate = 0.9995
    
    no_improve_count = 0
    last_improve_time = start_time
    
    while time.monotonic() < deadline:
        if T < T_min:
            break
            
        idx = random.randint(0, N - 1)
        old_val = current_set[idx]
        new_val = random.randint(1, MAX_VAL)
        
        if new_val == old_val:
            continue
            
        # Check distinctness
        # Using a set for membership check is O(1) but creating it is O(N).
        # Since N=32, `new_val in current_set` is O(32) comparisons.
        # This is very fast.
        if new_val in current_set:
            continue
            
        current_set[idx] = new_val
        new_score = get_score(current_set)
        
        delta = new_score - current_score
        
        if delta > 0:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
                last_improve_time = time.monotonic()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            if random.random() < math.exp(delta / T):
                current_score = new_score
                no_improve_count += 1
            else:
                current_set[idx] = old_val
                no_improve_count += 1
                
        T *= cooling_rate
        
        # Restart logic: if stuck for a long time, restart from best
        if time.monotonic() - last_improve_time > 10.0:
            # Restart from best
            current_set = best_set[:]
            current_score = best_score
            T = 50.0
            last_improve_time = time.monotonic()

    return best_set
