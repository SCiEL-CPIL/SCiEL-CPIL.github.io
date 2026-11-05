import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    k = 4
    
    all_sets = []
    for mask in range(1 << n):
        if bin(mask).count('1') == k:
            all_sets.append(mask)
            
    best_family = []
    best_size = 0
    
    while time.time() < deadline:
        random.shuffle(all_sets)
        family = []
        
        for s in all_sets:
            valid = True
            flen = len(family)
            for i in range(flen):
                a = family[i]
                for j in range(i + 1, flen):
                    b = family[j]
                    ab = a & b
                    if ab == (a & s) == (b & s):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(s)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
            if best_size >= 250:
                break

    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
