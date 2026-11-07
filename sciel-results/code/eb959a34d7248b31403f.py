import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_size = 0
    best_set = []
    
    def is_valid_addition(current_sums, x):
        new_sums = set(current_sums)
        if x in new_sums:
            return False, set()
        new_sums.add(x)
        for s in current_sums:
            ns = s + x
            if ns in new_sums:
                return False, set()
            new_sums.add(ns)
        
        # Check divisibility
        if len(new_sums) > 1000:
            max_sum = max(new_sums)
            present = [False] * (max_sum + 1)
            for s in new_sums:
                present[s] = True
            
            for s in new_sums:
                i = 1
                while i * i <= s:
                    if s % i == 0:
                        d1 = i
                        d2 = s // i
                        if d1 < s and present[d1]:
                            return False, set()
                        if d2 < s and d2 != d1 and present[d2]:
                            return False, set()
                    i += 1
        else:
            # Small set, do direct check
            sums_list = list(new_sums)
            for i in range(len(sums_list)):
                for j in range(i+1, len(sums_list)):
                    a, b = sums_list[i], sums_list[j]
                    if a < b:
                        if b % a == 0:
                            return False, set()
                    else:
                        if a % b == 0:
                            return False, set()
        
        return True, new_sums

    def greedy_search():
        nonlocal best_size, best_set
        elements = list(range(1, 257))
        random.shuffle(elements)
        
        current_elements = []
        current_sums = set()
        
        for x in elements:
            if time.time() > deadline:
                return
            valid, new_sums = is_valid_addition(current_sums, x)
            if valid:
                current_elements.append(x)
                current_sums = new_sums
                if len(current_elements) > best_size:
                    best_size = len(current_elements)
                    best_set = list(current_elements)
        
    while time.time() < deadline:
        greedy_search()
        if best_size >= 15 and time.time() > deadline - 5:
            break

    if not best_set:
        return [31, 32, 34]
    
    return best_set
