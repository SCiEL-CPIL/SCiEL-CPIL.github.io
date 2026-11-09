import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    base = [1, 2, 5, 11]
    best_set = base[:]
    best_size = len(base)
    
    def update_best(s):
        nonlocal best_set, best_size
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            if best_size == 20:
                return True
        return False

    def greedy_randomized():
        current_set = set(base)
        current_sums = set()
        lst = list(current_set)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                current_sums.add(lst[i] + lst[j])
        
        while True:
            if time.time() > deadline:
                return list(current_set)
            
            valid_candidates = []
            for x in range(1, 285):
                if x in current_set:
                    continue
                valid = True
                for y in current_set:
                    if x + y in current_sums:
                        valid = False
                        break
                if valid and 2 * x in current_sums:
                    valid = False
                if valid:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            x = random.choice(valid_candidates)
            
            current_set.add(x)
            for y in current_set:
                if y != x:
                    current_sums.add(x + y)
            current_sums.add(2 * x)
            
            if update_best(list(current_set)):
                return list(current_set)
        
        return list(current_set)

    def local_search_improve(initial_set):
        current_set = set(initial_set)
        current_sums = set()
        lst = list(current_set)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                current_sums.add(lst[i] + lst[j])
        
        while True:
            if time.time() > deadline:
                return list(current_set)
            
            added = False
            for x in range(1, 285):
                if x in current_set:
                    continue
                valid = True
                for y in current_set:
                    if x + y in current_sums:
                        valid = False
                        break
                if valid and 2 * x in current_sums:
                    valid = False
                if valid:
                    current_set.add(x)
                    for y in current_set:
                        if y != x:
                            current_sums.add(x + y)
                    current_sums.add(2 * x)
                    added = True
                    if update_best(list(current_set)):
                        return list(current_set)
                    break
            
            if not added:
                swap_done = False
                for _ in range(50):
                    if time.time() > deadline:
                        return list(current_set)
                    
                    y = random.choice(list(current_set))
                    temp_set = current_set - {y}
                    temp_sums = set()
                    t_lst = list(temp_set)
                    for i in range(len(t_lst)):
                        for j in range(i, len(t_lst)):
                            temp_sums.add(t_lst[i] + t_lst[j])
                    
                    valid_x = None
                    for x in range(1, 285):
                        if x in temp_set:
                            continue
                        if x == y:
                            continue
                        valid = True
                        for z in temp_set:
                            if x + z in temp_sums:
                                valid = False
                                break
                        if valid and 2 * x in temp_sums:
                            valid = False
                        if valid:
                            valid_x = x
                            break
                    
                    if valid_x is not None:
                        current_set = temp_set
                        current_set.add(valid_x)
                        current_sums = temp_sums
                        for z in current_set:
                            if z != valid_x:
                                current_sums.add(valid_x + z)
                        current_sums.add(2 * valid_x)
                        swap_done = True
                        break
                
                if not swap_done:
                    break
        
        return list(current_set)

    while time.time() < deadline:
        s = greedy_randomized()
        if len(s) >= best_size:
            if len(s) > best_size:
                update_best(s)
            if best_size == 20:
                return best_set
        
        s2 = local_search_improve(s)
        if len(s2) > best_size:
            update_best(s2)
            if best_size == 20:
                return best_set

    return best_set
