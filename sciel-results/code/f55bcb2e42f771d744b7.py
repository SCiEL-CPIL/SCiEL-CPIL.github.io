import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    all_masks = []
    for c in combinations(range(16), 4):
        m = 0
        for x in c:
            m |= (1 << x)
        all_masks.append(m)
    n_sets = len(all_masks)
    
    best_family = []
    best_size = 0
    
    def can_add(family, s):
        groups = {}
        for ai in family:
            t = ai & s
            if t not in groups:
                groups[t] = []
            groups[t].append(ai)
        for t, members in groups.items():
            if len(members) < 2:
                continue
            n = len(members)
            for x in range(n):
                ax = members[x]
                for y in range(x + 1, n):
                    if ax & members[y] == t:
                        return False
        return True
    
    for trial in range(50):
        if time.time() > deadline - 3:
            break
        
        rng = random.Random(trial * 7919 + 1)
        family = []
        
        # Greedy phase with random order
        order = list(range(n_sets))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 3:
                break
            s = all_masks[idx]
            if can_add(family, s):
                family.append(s)
        
        # Local search: sweep to add more sets
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            changed = False
            in_family = set(family)
            for idx in range(n_sets):
                if time.time() > deadline - 3:
                    break
                s = all_masks[idx]
                if s not in in_family and can_add(family, s):
                    family.append(s)
                    in_family.add(s)
                    changed = True
            if not changed:
                break
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    # Final polish
    if time.time() < deadline - 1:
        in_best = set(best_family)
        for idx in range(n_sets):
            if time.time() > deadline - 1:
                break
            s = all_masks[idx]
            if s not in in_best and can_add(best_family, s):
                best_family.append(s)
                in_best.add(s)
    
    result = []
    for m in best_family:
        elems = [x + 1 for x in range(16) if m & (1 << x)]
        result.append(elems)
    
    return result
