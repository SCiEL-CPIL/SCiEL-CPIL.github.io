from itertools import combinations
import random

def construct():
    all_sets = list(combinations(range(1, 17), 4))
    n = len(all_sets)
    masks = [0] * n
    for i, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        masks[i] = m
        
    def greedy_run(seed):
        random.seed(seed)
        available = set(range(n))
        chosen = []
        conflict_counts = [0] * n
        
        while available:
            valid = [k for k in available if conflict_counts[k] == 0]
            if not valid:
                break
            
            u = random.choice(valid)
            
            chosen.append(u)
            available.remove(u)
            
            # Update conflict counts for pairs (u, v) where v is in previous chosen
            for v in chosen[:-1]:
                i_u = masks[u]
                i_v = masks[v]
                inter = i_u & i_v
                union_diff = (i_u | i_v) ^ inter
                
                # Iterate over a list of available to avoid set iteration overhead
                avail_list = list(available)
                for w in avail_list:
                    i_w = masks[w]
                    if (i_w & inter) == inter and (i_w & union_diff) == 0:
                        conflict_counts[w] += 1
                        
        return [sorted([x for x in range(1, 17) if (masks[i] >> (x-1)) & 1]) for i in chosen]

    best_family = []
    best_len = 0
    
    # Run multiple restarts
    for s in range(100):
        family = greedy_run(s)
        if len(family) > best_len:
            best_len = len(family)
            best_family = family
            
    return best_family
