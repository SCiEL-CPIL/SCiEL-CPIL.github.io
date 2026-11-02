def construct():
    import math
    import random
    import time

    def score_of(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        diff_set = set()
        sum_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
                sum_set.add(a + b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
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
    
    best_candidate = None
    best_score = 0
    
    # Initial diverse pool
    initial_sets = [
        [0, 1, 3, 7, 15],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 13, 40],
        [0, 2, 5, 12, 29],
        [0, 1, 3, 10, 31],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 6, 12],
        [0, 1, 4, 10, 25],
        [0, 2, 6, 18],
        [0, 1, 3, 9],
        [0, 1, 4, 13],
        [0, 1, 3, 7, 13],
        [0, 1, 3, 7, 11],
        [0, 3, 8, 18, 39],
        [0, 1, 4, 16, 65],
        [0, 2, 7, 22, 67],
    ]
    
    for s in initial_sets:
        sc = score_of(s)
        if sc > best_score:
            best_score = sc
            best_candidate = s[:]

    # Local Search with Random Restarts
    start_time = time.time()
    time_limit = 115  # Seconds

    while time.time() - start_time < time_limit:
        # Generate a new starting point
        if random.random() < 0.3 and best_candidate:
            # Perturb best solution
            current = best_candidate[:]
            # Randomly swap one element
            if len(current) > 3:
                idx = random.randint(1, len(current)-1)
                current[idx] = random.randint(1, 300)
                current = sorted(set(current))
                if len(current) != len(best_candidate):
                    current = best_candidate[:]
        else:
            # Generate random set
            n = random.randint(4, 8)
            max_val = random.randint(20, 300)
            elems = random.sample(range(1, max_val+1), n-1)
            current = sorted([0] + elems)
        
        current_score = score_of(current)
        
        # Hill Climb
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try 100 random moves
            for _ in range(100):
                move_type = random.choice(['replace', 'add', 'remove', 'swap'])
                
                if move_type == 'replace' and len(current) > 3:
                    idx = random.randint(1, len(current)-1)
                    old_val = current[idx]
                    new_val = random.randint(1, 300)
                    if new_val == old_val or new_val in current:
                        continue
                    new_set = current[:]
                    new_set[idx] = new_val
                    new_set = sorted(new_set)
                    ns = score_of(new_set)
                    if ns > current_score:
                        current = new_set
                        current_score = ns
                        improved = True
                        break

                elif move_type == 'add' and len(current) < 9:
                    new_val = random.randint(1, 300)
                    if new_val in current:
                        continue
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

                elif move_type == 'swap' and len(current) > 4:
                    i, j = random.sample(range(1, len(current)), 2)
                    new_set = current[:]
                    new_set[i], new_set[j] = new_set[j], new_set[i]
                    new_set = sorted(new_set)
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
