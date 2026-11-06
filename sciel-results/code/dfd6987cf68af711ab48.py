import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    limit = 512
    
    def can_add(x, S_set, S_list):
        for y in S_list:
            if x * y // gcd(x, y) > limit:
                return False
        return True

    # Initial set
    S_list = list(range(1, 23)) + [24, 26]
    S_set = set(S_list)
    
    best_S_list = list(S_list)
    
    def try_add(S_list, S_set):
        for x in range(1, limit + 1):
            if x not in S_set and can_add(x, S_set, S_list):
                new_list = S_list + [x]
                new_set = S_set | {x}
                return new_list, new_set
        return None, None

    def try_swap(S_list, S_set):
        best_size = len(S_list)
        best_new_list = None
        best_new_set = None
        
        for y in S_list:
            # Remove y
            S_temp_list = [z for z in S_list if z != y]
            S_temp_set = S_set - {y}
            
            # Greedy add to S_temp
            added_any = False
            for x in range(1, limit + 1):
                if x not in S_temp_set and can_add(x, S_temp_set, S_temp_list):
                    S_temp_list.append(x)
                    S_temp_set.add(x)
                    added_any = True
            
            if added_any and len(S_temp_list) > best_size:
                best_size = len(S_temp_list)
                best_new_list = list(S_temp_list)
                best_new_set = set(S_temp_set)
                
        return best_new_list, best_new_set

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try add
        new_list, new_set = try_add(S_list, S_set)
        if new_list is not None:
            S_list = new_list
            S_set = new_set
            improved = True
            if len(S_list) > len(best_S_list):
                best_S_list = list(S_list)
            continue
        
        # Try swap
        new_list, new_set = try_swap(S_list, S_set)
        if new_list is not None:
            S_list = new_list
            S_set = new_set
            improved = True
            if len(S_list) > len(best_S_list):
                best_S_list = list(S_list)
            continue
            
    return best_S_list
