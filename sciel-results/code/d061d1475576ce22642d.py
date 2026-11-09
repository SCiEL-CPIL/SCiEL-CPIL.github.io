import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin
    
    max_val = 4096
    
    best_set = []
    best_size = 0
    best_sum_set = set()
    
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

    # 2. Randomized Greedy
    rng = random.Random(42)
    # Run random passes until time is low
    for i in range(100):
        if time.time() > deadline - 15.0: # Leave 15s for local search
            break
            
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        s, ss = run_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sum_set = ss

    # 3. Simulated Annealing / Local Search
    if time.time() < deadline - 2.0:
        S = list(best_set)
        sum_set = set(best_sum_set)
        size = len(S)
        
        # Helper to remove element
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

        # Helper to add element
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

        # Simulated Annealing
        # State: S (list), sum_set (set), S_set (set of elements in S)
        # Moves: Add random element, Remove random element, Swap random element with random other
        
        S_set = set(S)
        current_size = len(S)
        
        # Initial temperature
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.995
        
        # We want to maximize size. 
        # "Energy" is -size. 
        # If we increase size, delta_E < 0, always accept.
        # If we decrease size, delta_E > 0, accept with prob exp(-delta_E/T).
        
        steps = 0
        max_steps = 50000
        
        while time.time() < deadline - 1.0:
            move_type = rng.random()
            
            # 30% Add, 20% Remove, 50% Swap
            if move_type < 0.3:
                # Try Add
                c = rng.randint(1, max_val)
                if c in S_set:
                    continue
                
                S_backup = list(S)
                sum_set_backup = set(sum_set)
                S_set_backup = set(S_set)
                
                if add_element(c, S, sum_set):
                    S_set.add(c)
                    # Success: Size increased by 1
                    # Always accept (improvement)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = list(S)
                        best_sum_set = set(sum_set)
                else:
                    # Revert
                    S = S_backup
                    sum_set = sum_set_backup
                    S_set = S_set_backup
                    
            elif move_type < 0.5:
                # Try Remove
                if not S:
                    continue
                a = rng.choice(S)
                
                S_backup = list(S)
                sum_set_backup = set(sum_set)
                S_set_backup = set(S_set)
                
                remove_element(a, S, sum_set)
                S_set.remove(a)
                
                # Size decreased by 1. 
                # This is a "worse" move in terms of objective (maximize size).
                # Accept with probability exp(-1/T)
                if rng.random() < (2.718281828 ** (-1.0 / T)):
                    current_size -= 1
                    # Keep the change
                else:
                    # Revert
                    S = S_backup
                    sum_set = sum_set_backup
                    S_set = S_set_backup
                    
            else:
                # Try Swap
                if not S:
                    continue
                a = rng.choice(S)
                b = rng.randint(1, max_val)
                if b in S_set:
                    continue
                
                S_backup = list(S)
                sum_set_backup = set(sum_set)
                S_set_backup = set(S_set)
                
                remove_element(a, S, sum_set)
                S_set.remove(a)
                
                if add_element(b, S, sum_set):
                    S_set.add(b)
                    # Size unchanged. 
                    # Accept with probability 1 (neutral move, helps exploration)
                    # Or we could use a slight bias, but neutral is fine for SA exploration
                else:
                    # Revert
                    S = S_backup
                    sum_set = sum_set_backup
                    S_set = S_set_backup
            
            steps += 1
            T *= cooling_rate
            if T < T_min:
                T = T_min

    return best_set
