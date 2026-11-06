"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_subsets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_subsets.append(mask)
    
    def is_sunflower(a, b, c):
        return (a & b) == (a & c) == (b & c)
    
    def try_greedy(order):
        family = []
        for mask in order:
            ok = True
            for i in range(len(family)):
                for j in range(i+1, len(family)):
                    if is_sunflower(family[i], family[j], mask):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                family.append(mask)
        return family
    
    best = []
    
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        shuffled = all_subsets[:]
        random.shuffle(shuffled)
        fam = try_greedy(shuffled)
        if len(fam) > len(best):
            best = fam
    
    if time.time() < deadline - 3:
        best_set = set(best)
        for mask in all_subsets:
            if mask not in best_set:
                ok = True
                for i in range(len(best)):
                    for j in range(i+1, len(best)):
                        if is_sunflower(best[i], best[j], mask):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best.append(mask)
                    best_set.add(mask)
    
    result = []
    for mask in best:
        elements = [i+1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))
    return result
# EVOLVE-BLOCK-END
