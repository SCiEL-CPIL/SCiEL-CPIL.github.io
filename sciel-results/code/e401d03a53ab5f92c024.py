import random
import time

def _validate(elements):
    n = len(elements)
    if n == 0:
        return True
    
    sums = set()
    num_subsets = 1 << n
    for mask in range(1, num_subsets):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
        
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b % a == 0:
                return False
    return True

def construct():
    deadline = time.time() + 85
    best_set = [31, 32, 34]
    best_size = 3
    
    if _validate(best_set):
        best_size = len(best_set)
        
    random.seed(42)
    
    # Try to find larger sets using greedy approach with random restarts
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Start with a random large number
        start_val = random.randint(100, 256)
        current = [start_val]
        sums = {start_val}
        max_sum = start_val
        
        # Try to add more elements
        candidates = [x for x in range(1, 257) if x not in current and x != start_val]
        # Prefer larger numbers
        candidates.sort(reverse=True)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
                
            # Generate new sums
            new_sums = set()
            valid = True
            for s in sums:
                ns = s + x
                if ns in sums:
                    valid = False
                    break
                new_sums.add(ns)
            new_sums.add(x)
            
            if not valid:
                continue
                
            if x in sums:
                continue
                
            # Optimization for x > max_sum
            if x > max_sum:
                # S_new elements are all > max_sum
                # So S cannot be divided by S_new (S < S_new)
                # Only check if S divides S_new
                # and if S_new divides S_new
                
                # Check S divides S_new
                for s in sums:
                    for ns in new_sums:
                        if ns % s == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                    
                # Check S_new divides S_new
                new_sums_list = sorted(new_sums)
                for i in range(len(new_sums_list)):
                    a = new_sums_list[i]
                    for j in range(i + 1, len(new_sums_list)):
                        b = new_sums_list[j]
                        if b % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
            else:
                # General case
                for s in sums:
                    for ns in new_sums:
                        if ns > s:
                            if ns % s == 0:
                                valid = False
                                break
                        elif ns < s:
                            if s % ns == 0:
                                valid = False
                                break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_sums_list = sorted(new_sums)
                for i in range(len(new_sums_list)):
                    a = new_sums_list[i]
                    for j in range(i + 1, len(new_sums_list)):
                        b = new_sums_list[j]
                        if b % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                        
            if not valid:
                continue
                
            current.append(x)
            sums.update(new_sums)
            max_sum = max(max_sum, max(new_sums))
            
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    return list(best_set)
