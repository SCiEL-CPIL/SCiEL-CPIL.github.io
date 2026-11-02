def construct():
    import math
    import random
    import time
    import numpy as np

    def score_of(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        U_arr = np.array(U, dtype=int)
        sum_set = np.add.outer(U_arr, U_arr)
        diff_set = np.subtract.outer(U_arr, U_arr)
        
        # Flatten and unique
        n_sum = len(np.unique(sum_set))
        n_diff = len(np.unique(diff_set))
        
        if n_sum == 0 or n_diff == 0:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1
        return 1 + math.log(ratio) / denom

    random.seed(42)
    
    def greedy_golomb(n):
        U = [0]
        diffs = set()
        for _ in range(n - 1):
            x = U[-1] + 1
            while True:
                new_diffs = [x - u for u in U]
                if not any(d in diffs for d in new_diffs):
                    U.append(x)
                    diffs.update(new_diffs)
                    break
                x += 1
        return U

    initial_sets = []
    for n in range(3, 10):
        initial_sets.append(greedy_golomb(n))
        
    # Also add some random perturbations of these
    for base in initial_sets:
        for _ in range(3):
            perturbed = base[:]
            if len(perturbed) > 1:
                idx = random.randint(1, len(perturbed)-1)
                perturbed[idx] += random.randint(-2, 2)
                if perturbed[idx] < 0:
                    perturbed[idx] = 0
                perturbed = sorted(set(perturbed))
            initial_sets.append(perturbed)

    best_candidate = None
    best_score = 0
    
    for s in initial_sets:
        sc = score_of(s)
        if sc > best_score:
            best_score = sc
            best_candidate = s[:]

    start_time = time.time()
    time_limit = 115

    while time.time() - start_time < time_limit:
        if random.random() < 0.5 and best_candidate:
            current = best_candidate[:]
            # Perturb
            if len(current) > 3:
                idx = random.randint(1, len(current)-1)
                current[idx] += random.randint(-5, 5)
                if current[idx] < 0:
                    current[idx] = 0
                current = sorted(set(current))
        else:
            # Random start
            n = random.randint(4, 8)
            max_val = random.randint(20, 100)
            elems = random.sample(range(1, max_val+1), n-1)
            current = sorted([0] + elems)
            
        current_score = score_of(current)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for _ in range(50):
                move_type = random.choice(['replace', 'add', 'remove'])
                
                if move_type == 'replace' and len(current) > 3:
                    idx = random.randint(1, len(current)-1)
                    new_val = current[idx] + random.randint(-10, 10)
                    if new_val < 0: continue
                    if new_val in current: continue
                    new_set = current[:]
                    new_set[idx] = new_val
                    new_set = sorted(new_set)
                    ns = score_of(new_set)
                    if ns > current_score:
                        current = new_set
                        current_score = ns
                        improved = True
                        break

                elif move_type == 'add' and len(current) < 10:
                    new_val = random.randint(1, max(current)+10 if current else 10)
                    if new_val in current: continue
                    new_set = sorted(current + [new_val])
                    ns = score_of(new_set)
                    if ns > current_score:
                        current = new_set
                        current_score = ns
                        improved = True
                        break

                elif move_type == 'remove' and len(current) > 4:
                    idx = random.randint(1, len(current)-1)
                    new_set = current[:idx] + current[idx+1:]
                    ns = score_of(new_set)
                    if ns > current_score:
                        current = new_set
                        current_score = ns
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_candidate = current[:]

    return best_candidate
