import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_optimize(V, time_limit):
        S = set(V)
        removed = list(V)
        
        # 1. Greedy Removal with efficient conflict detection
        while not is_sum_free(S):
            if time.time() > start_time + time_limit:
                break
            
            # Count conflicts efficiently
            conflicts = {}
            S_list = list(S)
            S_set = S
            n = len(S_list)
            for i in range(n):
                a = S_list[i]
                for j in range(i, n):
                    b = S_list[j]
                    s = a + b
                    if s in S_set:
                        conflicts[a] = conflicts.get(a, 0) + 1
                        if b != a:
                            conflicts[b] = conflicts.get(b, 0) + 1
                        conflicts[s] = conflicts.get(s, 0) + 1
            
            if not conflicts:
                break
                
            # Pick element with maximum conflicts
            max_deg = max(conflicts.values())
            candidates = [x for x, c in conflicts.items() if c == max_deg]
            x_to_remove = random.choice(candidates)
            
            S.remove(x_to_remove)
            removed.remove(x_to_remove)

        # 2. Hill Climbing with improved can_add check
        def can_add(x, S_set, S_list):
            # Check if x conflicts with any pair in S
            # Conflict conditions:
            # 1. x + a = b for some a,b in S (x is "sum")
            # 2. a + x = b for some a,b in S (x is "addend")  
            # 3. a + b = x for some a,b in S (x is "sum")
            # All covered by: for any a in S, check if x+a in S OR x-a in S
            for a in S_list:
                if (x + a) in S_set or (x - a) in S_set:
                    return False
            return True

        # Multiple passes of hill climbing
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_removed = []
            random.shuffle(removed)
            S_list = list(S)
            S_set = S
            
            for x in removed:
                if time.time() > start_time + time_limit:
                    new_removed.append(x)
                    continue
                
                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                else:
                    new_removed.append(x)
            
            if improved:
                removed = new_removed
        
        return list(S)

    results = []
    
    # Process instances with adaptive restarts based on remaining time
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        best_size = 0
        best_set = []
        
        # Adaptive restarts: start with 1, add more if time permits
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        
        # Allocate time: first restart gets more, subsequent get less
        # For small instances, we can afford more restarts
        if size <= 60:
            max_restarts = 3
            base_time = 8.0
        elif size <= 120:
            max_restarts = 2
            base_time = 12.0
        else:
            max_restarts = 2
            base_time = 20.0
        
        for r in range(max_restarts):
            if time.time() > start_time + DEADLINE - 5:
                break
            
            # Decreasing time per restart
            time_per_restart = base_time * (0.7 ** r)
            
            res = greedy_optimize(V, time_per_restart)
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
            
            # Check if we should continue
            if time.time() > start_time + DEADLINE - 10:
                break
        
        results.append(best_set)

    return results
