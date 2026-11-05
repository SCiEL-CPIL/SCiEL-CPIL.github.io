import random
import time

def score_set(S):
    S_set = set(S)
    good_d = set()
    for d in range(1, 512):
        for a in S:
            if (a + d) in S_set and (a + 2 * d) in S_set:
                good_d.add(d)
                break
    return len(good_d)

def greedy_construct(K_target, max_elements=32):
    """
    Construct a set S that tries to cover differences 1..K_target.
    """
    S = set()
    for d in range(1, K_target + 1):
        if len(S) >= max_elements:
            break
        
        # Check if d is already covered
        covered = False
        for a in S:
            if (a + d) in S and (a + 2 * d) in S:
                covered = True
                break
        
        if not covered:
            # Try to find a spot to add elements to cover d
            # We prefer to reuse existing elements if possible
            # Try all a in 1..512
            best_a = None
            min_new_elements = max_elements + 1
            
            for a in range(1, 513):
                req = [a, a + d, a + 2 * d]
                if req[2] > 512:
                    continue
                
                new_needed = [x for x in req if x not in S]
                if len(new_needed) < min_new_elements:
                    min_new_elements = len(new_needed)
                    best_a = a
                    
                    if min_new_elements == 0:
                        break
            
            if best_a is not None:
                req = [best_a, best_a + d, best_a + 2 * d]
                for x in req:
                    if len(S) < max_elements:
                        S.add(x)
                    else:
                        break
                        
    # If we didn't fill 32 elements, fill with randoms or extend
    while len(S) < max_elements:
        x = random.randint(1, 512)
        if x not in S:
            S.add(x)
            
    return list(S)

def run_search(initial_S, time_budget):
    start_time = time.time()
    S = initial_S.copy()
    current_score = score_set(S)
    best_S = S.copy()
    best_score = current_score
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_budget:
        new_S = S.copy()
        idx = random.randint(0, 31)
        old_val = new_S[idx]
        while True:
            new_val = random.randint(1, 512)
            if new_val not in new_S:
                break
        new_S[idx] = new_val
        
        new_score = score_set(new_S)
        
        if new_score > current_score:
            S = new_S
            current_score = new_score
            if current_score > best_score:
                best_S = S.copy()
                best_score = current_score
        else:
            delta = current_score - new_score
            if delta == 0:
                accept = True
            else:
                accept = random.random() < pow(2.71828, -delta / T)
            
            if accept:
                S = new_S
                current_score = new_score
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_S, best_score

def construct():
    start_time = time.time()
    total_time = 115
    
    best_S = []
    best_score = 0
    
    # 1. Greedy Construction for various K
    # Try to cover d=1..20, d=1..30, etc.
    for K in [10, 15, 20, 25, 30]:
        S_greedy = greedy_construct(K)
        score_greedy = score_set(S_greedy)
        if score_greedy > best_score:
            best_S = S_greedy
            best_score = score_greedy
            
    # 2. Refine best greedy result with SA
    time_remaining = total_time - (time.time() - start_time)
    if time_remaining > 10:
        time_per_run = time_remaining / 4
        S_refined, score_refined = run_search(best_S, time_per_run)
        if score_refined > best_score:
            best_S = S_refined
            best_score = score_refined
            
    # 3. Random Starts
    for i in range(3):
        if time.time() - start_time > total_time - 5:
            break
        random_S = random.sample(range(1, 513), 32)
        S_rand, score_rand = run_search(random_S, time_per_run)
        if score_rand > best_score:
            best_S = S_rand
            best_score = score_rand
            
    return best_S
