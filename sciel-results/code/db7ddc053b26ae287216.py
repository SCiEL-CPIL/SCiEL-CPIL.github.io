import time
import random
import math

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We target 115s to be safe.
    time_limit = 115.0
    
    def get_score(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        max_u = max(U)
        if max_u == 0:
            return 1.0
        denom = math.log(2 * max_u + 1)
        if denom < 1e-9:
            return 1.0
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 1.0
        return 1 + math.log(ratio) / denom

    # Initial best known from analysis
    best_set = [0, 1, 3, 9, 27]
    best_score = get_score(best_set)
    
    # Search parameters
    max_n = 7
    restarts_per_n = 10
    steps_per_restart = 200
    
    for n in range(3, max_n + 1):
        if time.time() - start_time > time_limit - 10:
            break
            
        for r in range(restarts_per_n):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Initialize
            if r == 0:
                U = [3**i for i in range(n)]
            else:
                U = [0]
                last = 0
                for _ in range(n - 1):
                    # Random next element
                    # To avoid huge max U, cap it
                    # But we want to explore. 
                    # Let's pick from a range that allows good ratios.
                    # Powers of 3 grow fast. 
                    # Let's try random integers.
                    while True:
                        cand = random.randint(last + 1, 100)
                        if cand not in U:
                            U.append(cand)
                            last = cand
                            break
            U = sorted(set(U))
            if len(U) != n:
                # Fix size
                if len(U) < n:
                    last = U[-1] if U else 0
                    while len(U) < n:
                        last += 1
                        if last not in U:
                            U.append(last)
                    U = sorted(U)
                else:
                    U = U[:n]
                    
            cur_score = get_score(U)
            
            # Local Search
            for step in range(steps_per_restart):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                i = random.randint(0, n - 1)
                old_val = U[i]
                
                # Perturbation
                # Mix of local and global moves
                if random.random() < 0.7:
                    delta = random.randint(-5, 5)
                else:
                    delta = random.randint(-50, 50)
                    
                new_val = old_val + delta
                if new_val < 0 or new_val in U:
                    continue
                    
                U[i] = new_val
                U = sorted(U)
                
                new_score = get_score(U)
                
                if new_score > cur_score:
                    cur_score = new_score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = U.copy()
                else:
                    # Revert
                    U[i] = old_val
                    U = sorted(U)

    return best_set
