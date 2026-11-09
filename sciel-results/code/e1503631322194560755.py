import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin
    
    max_val = 4096
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(x, current_set, sum_set):
        new_sums = [3 * x]
        for y in current_set:
            new_sums.append(2 * x + y)
            new_sums.append(x + 2 * y)
        for y, z in combinations(current_set, 2):
            new_sums.append(x + y + z)
        
        new_sums_set = set(new_sums)
        if len(new_sums_set) != len(new_sums):
            return False, None
        if not new_sums_set.isdisjoint(sum_set):
            return False, None
        return True, new_sums_set

    def run_greedy(candidates):
        current_set = []
        sum_set = set()
        
        for x in candidates:
            valid, new_sums = is_valid_addition(x, current_set, sum_set)
            if valid:
                current_set.append(x)
                sum_set.update(new_sums)
                
        return current_set, sum_set

    # 1. Deterministic Increasing Greedy
    s, ss = run_greedy(range(1, max_val + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        best_sum_set = ss

    # 2. Randomized Greedy with multiple seeds
    rng = random.Random(42)
    for i in range(40):
        if time.time() > deadline - 12.0:
            break
            
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        s, ss = run_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sum_set = ss

    # 3. Local Search (Swap and Add)
    if time.time() < deadline - 5.0:
        S = list(best_set)
        sum_set = best_sum_set
        
        def remove_element(x, S, sum_set):
            to_remove = set()
            to_remove.add(3*x)
            for y in S:
                if y == x: continue
                to_remove.add(2*x + y)
                to_remove.add(x + 2*y)
            for y, z in combinations(S, 2):
                if y == x or z == x: continue
                to_remove.add(x + y + z)
            
            sum_set.difference_update(to_remove)
            S.remove(x)

        def add_element(x, S, sum_set):
            new_sums = [3 * x]
            for y in S:
                new_sums.append(2 * x + y)
                new_sums.append(x + 2 * y)
            for y, z in combinations(S, 2):
                new_sums.append(x + y + z)
            
            new_sums_set = set(new_sums)
            if len(new_sums_set) != len(new_sums):
                return False
            if not new_sums_set.isdisjoint(sum_set):
                return False
            
            sum_set.update(new_sums_set)
            S.append(x)
            return True

        # Try to add elements directly first
        for c in range(1, max_val + 1):
            if c in S:
                continue
            if add_element(c, S, sum_set):
                pass  # element added
            if time.time() > deadline - 2.0:
                break
        
        # Local Search Loop with more aggressive strategy
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try random swaps with multiple attempts per round
            for _ in range(100):
                if time.time() > deadline - 2.0:
                    break
                
                if len(S) == 0: 
                    break
                a = rng.choice(S)
                b = rng.randint(1, max_val)
                if b in S:
                    continue
                
                # Try swap a -> b
                S_backup = list(S)
                sum_set_backup = set(sum_set)
                
                remove_element(a, S, sum_set)
                
                if add_element(b, S, sum_set):
                    # Swap successful. Try to add multiple c's
                    added_count = 0
                    for c in range(1, max_val + 1):
                        if c in S: continue
                        S_backup2 = list(S)
                        sum_set_backup2 = set(sum_set)
                        
                        if add_element(c, S, sum_set):
                            added_count += 1
                            improved = True
                            if time.time() > deadline - 2.0:
                                break
                        else:
                            S = S_backup2
                            sum_set = sum_set_backup2
                    
                    if added_count == 0:
                        # Swap didn't lead to growth, revert swap
                        S = S_backup
                        sum_set = sum_set_backup
                else:
                    # Swap invalid, revert
                    S = S_backup
                    sum_set = sum_set_backup
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set
