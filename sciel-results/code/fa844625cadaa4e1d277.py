import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)
    
    inter = [[0]*n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_sets[i]
        for j in range(i+1, n_total):
            ic = bin(mi & all_sets[j]).count('1')
            inter[i][j] = ic
            inter[j][i] = ic
    
    def can_add(idx, family):
        for a in family:
            ia = inter[idx][a]
            for b in family:
                if b <= a:
                    continue
                if inter[idx][b] == ia == inter[a][b]:
                    return False
        return True
    
    def greedy_randomized(order):
        chosen = set()
        for idx in order:
            if can_add(idx, chosen):
                chosen.add(idx)
        return chosen
    
    def local_search(chosen, time_limit=1.0):
        chosen = set(chosen)
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            for add in range(n_total):
                if add not in chosen and can_add(add, chosen):
                    chosen.add(add)
                    improved = True
                    break
            if not improved:
                for rem in list(chosen):
                    remaining = chosen - {rem}
                    for add in range(n_total):
                        if add not in chosen and can_add(add, remaining):
                            chosen = remaining | {add}
                            improved = True
                            break
                    if improved:
                        break
        return chosen
    
    best = set()
    best_size = 0
    
    for r in range(500):
        if time.time() > deadline - 2.0:
            break
        order = list(range(n_total))
        random.shuffle(order)
        chosen = greedy_randomized(order)
        chosen = local_search(chosen, time_limit=0.8)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)
    
    result = []
    for mask in best:
        elements = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elements))
    return result
