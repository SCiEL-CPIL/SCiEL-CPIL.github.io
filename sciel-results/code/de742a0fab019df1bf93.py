import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    def is_valid(s):
        """
        Check if set s violates the condition:
        No three distinct members a, b, c in s such that a+b, a+c, b+c are all in s.
        """
        s_set = set(s)
        s_list = sorted(s)
        n = len(s_list)
        
        if n < 3:
            return True
        
        # For each triple of distinct elements (a, b, c) with a < b < c:
        # Check if a+b in s, a+c in s, b+c in s
        for i in range(n):
            a = s_list[i]
            for j in range(i+1, n):
                b = s_list[j]
                ab = a + b
                if ab not in s_set:
                    continue
                for k in range(j+1, n):
                    c = s_list[k]
                    ac = a + c
                    bc = b + c
                    if ac in s_set and bc in s_set:
                        return False
        return True

    def greedy_construct(order):
        """
        Greedy construction using a specific order of candidates.
        """
        s = set()
        for x in order:
            trial = s.union([x])
            if is_valid(trial):
                s = trial
        return s

    def local_search(s, deadline):
        """
        Local search: try adding elements, then swapping.
        """
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each element
            for x in range(1, 129):
                if time.time() > deadline:
                    break
                if x not in s:
                    trial = s.union([x])
                    if is_valid(trial):
                        s = trial
                        improved = True
                        break  # Restart loop to check new additions
            
            # Try removing one and adding one (swap)
            if not improved:
                for x in list(s):
                    if time.time() > deadline:
                        break
                    for y in range(1, 129):
                        if time.time() > deadline:
                            break
                        if y not in s:
                            trial = s.difference([x]).union([y])
                            if is_valid(trial):
                                s = trial
                                improved = True
                                break
                    if improved:
                        break
        
        return s

    def remove_and_add_two(s, deadline):
        """
        Try removing one element and adding two new ones.
        """
        s = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for rem in sorted(s):
                if time.time() > deadline:
                    break
                trial_base = s - {rem}
                # Try to add elements
                for add1 in range(1, 129):
                    if time.time() > deadline:
                        break
                    if add1 in trial_base or add1 == rem:
                        continue
                    trial1 = trial_base | {add1}
                    if is_valid(trial1):
                        trial_base = trial1
                        # Now try to add a second element
                        for add2 in range(1, 129):
                            if time.time() > deadline:
                                break
                            if add2 in trial_base or add2 == rem or add2 == add1:
                                continue
                            trial2 = trial_base | {add2}
                            if is_valid(trial2):
                                trial_base = trial2
                                break
                        if len(trial_base) > len(s):
                            s = trial_base
                            improved = True
                            break
                if improved:
                    break
        return s

    # Strategy:
    # 1. Start with the seed solution (65..128)
    # 2. Try greedy construction with different orders
    # 3. Local search to improve
    # 4. Try remove-and-add-two to break local optima
    
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    # Greedy with natural order (1..128)
    if time.time() < deadline - 10:
        s = greedy_construct(range(1, 129))
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Greedy with reverse order (128..1)
    if time.time() < deadline - 10:
        s = greedy_construct(range(128, 0, -1))
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Local search on best
    if time.time() < deadline - 5:
        improved_set = local_search(best_set, deadline - 5)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Try remove-and-add-two
    if time.time() < deadline - 5:
        improved_set = remove_and_add_two(best_set, deadline - 5)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Random restarts
    random.seed(42)
    while time.time() < deadline - 5:
        # Random order
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        s = greedy_construct(candidates)
        s = local_search(s, deadline - 5)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        
        # Try remove-and-add-two on the new result
        if time.time() < deadline - 5:
            improved_set = remove_and_add_two(s, deadline - 5)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
    
    # Final local search
    if time.time() < deadline - 2:
        improved_set = local_search(best_set, deadline - 2)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Final check: make sure the result is valid
    if not is_valid(best_set):
        # Fallback to seed
        best_set = set(range(65, 129))
    
    # Return as list of Python ints
    return [int(x) for x in sorted(best_set)]
