import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    all_masks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    n = len(all_masks)
    
    mask_to_elems = {m: tuple(i+1 for i in range(16) if (m >> i) & 1) for m in all_masks}
    
    best = []
    
    for restart in range(10000):
        if time.time() > deadline:
            break
        
        order = list(range(n))
        random.shuffle(order)
        
        family = []
        inter_map = {}
        
        for idx in order:
            if time.time() > deadline:
                break
            m = all_masks[idx]
            ok = True
            for a in family:
                ma = all_masks[a]
                I = ma & m
                for b in inter_map[a].get(I, ()):
                    if (m & all_masks[b]) == I:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                family.append(idx)
                inter_map[idx] = {}
                for a in family[:-1]:
                    ma = all_masks[a]
                    I = ma & m
                    if I not in inter_map[a]:
                        inter_map[a][I] = set()
                    inter_map[a][I].add(idx)
                    if I not in inter_map[idx]:
                        inter_map[idx][I] = set()
                    inter_map[idx][I].add(a)
        
        if len(family) > len(best):
            best = family[:]
    
    if not best:
        return []
    
    return [list(mask_to_elems[all_masks[i]]) for i in best]
