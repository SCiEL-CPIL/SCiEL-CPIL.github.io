import random
import time
from math import lcm, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Candidate numbers
    candidates = list(range(2, 257))
    
    def is_valid_addition(x, current_set):
        for y in current_set:
            if lcm(x, y) <= 256:
                return False
        return True
        
    def get_score(s):
        return sum(1.0/x for x in s)
        
    def local_search(initial_set):
        current_set = list(initial_set)
        current_score = get_score(current_set)
        current_set_set = set(current_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding
            for x in candidates:
                if x in current_set_set:
                    continue
                if is_valid_addition(x, current_set):
                    new_set = current_set + [x]
                    new_score = current_score + 1.0/x
                    if new_score > current_score:
                        current_set = new_set
                        current_set_set = set(current_set)
                        current_score = new_score
                        improved = True
            
            # Try swapping
            # To avoid O(N^2) swaps being too slow, we can limit or optimize
            # But N is small (max ~100), so N^2 is 10000 checks.
            # Each check is O(N) LCMs. 10000 * 100 = 1e6 LCMs. Fast.
            
            # We need to be careful: if we add, we might enable more adds.
            # The loop above handles that by restarting 'improved' logic.
            
            # Swap logic:
            for i in range(len(current_set)):
                for j in range(len(candidates)):
                    y = candidates[j]
                    if y in current_set_set:
                        continue
                    x = current_set[i]
                    if x == y:
                        continue
                    
                    # Check validity of new set
                    # Remove x, add y
                    # Check y against all others
                    valid = True
                    for k in range(len(current_set)):
                        if k == i:
                            continue
                        if lcm(y, current_set[k]) <= 256:
                            valid = False
                            break
                    
                    if valid:
                        new_score = current_score - 1.0/x + 1.0/y
                        if new_score > current_score + 1e-9:
                            # Perform swap
                            current_set[i] = y
                            current_set_set.remove(x)
                            current_set_set.add(y)
                            current_score = new_score
                            improved = True
        
        return current_set, current_score

    best_set = []
    best_score = 0.0
    
    # Generate several initial greedy sets
    # 1. Greedy starting with 2
    # 2. Greedy starting with 3
    # 3. Greedy starting with 4
    # 4. Random restarts
    
    def greedy_start(forbidden_start=None):
        s = []
        s_set = set()
        # Sort by 1/x descending (i.e., x ascending)
        for x in sorted(candidates):
            if forbidden_start is not None and x == forbidden_start:
                continue
            if is_valid_addition(x, s):
                s.append(x)
                s_set.add(x)
        return s

    # Run greedy for first few numbers
    for start_val in [2, 3, 4, 5, 6]:
        init = greedy_start()
        # Actually, greedy_start above doesn't force start_val, it just picks smallest valid.
        # To force start, we can pre-fill.
        
        # Let's just run the local search from a few random or structured starts
        pass

    # Better: Run local search from the standard greedy set
    init_greedy = greedy_start()
    s, score = local_search(init_greedy)
    if score > best_score:
        best_score = score
        best_set = s

    # Try starting with specific small numbers
    for forced in [2, 3, 4, 5]:
        init = [forced]
        init_set = {forced}
        # Greedy fill
        for x in sorted(candidates):
            if x in init_set:
                continue
            if is_valid_addition(x, init):
                init.append(x)
                init_set.add(x)
        
        s, score = local_search(init)
        if score > best_score:
            best_score = score
            best_set = s
            
    # Random restarts
    while time.time() < deadline - 5:
        # Random subset of small numbers
        random.shuffle(candidates)
        init = []
        for x in candidates[:20]:
            if is_valid_addition(x, init):
                init.append(x)
        
        s, score = local_search(init)
        if score > best_score:
            best_score = score
            best_set = s

    return best_set
