import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    k = 4
    
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    total_sets = len(all_sets)
    
    def is_sunflower(a, b, c):
        return (a & b) == (a & c) == (b & c)
    
    best_family = []
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        order = list(range(total_sets))
        random.shuffle(order)
        
        family = []
        
        for idx in order:
            if time.time() > deadline:
                break
            s = all_sets[idx]
            conflict = False
            for i in range(len(family)):
                a = family[i]
                for j in range(i+1, len(family)):
                    b = family[j]
                    if is_sunflower(s, a, b):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                family.append(s)
        
        if len(family) > len(best_family):
            best_family = family[:]
    
    result = []
    for mask in best_family:
        elements = [i for i in range(n) if (mask >> i) & 1]
        result.append(sorted(elements))
    return result
