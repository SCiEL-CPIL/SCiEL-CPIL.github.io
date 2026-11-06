import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    masks = []
    mask_to_set = {}
    for s in all_sets:
        m = 0
        for x in s:
            m |= (1 << (x-1))
        masks.append(m)
        mask_to_set[m] = s
    
    inter_mask = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        mi = masks[i]
        for j in range(i+1, n_sets):
            v = mi & masks[j]
            inter_mask[i][j] = v
            inter_mask[j][i] = v
    
    conflict_pairs = [set() for _ in range(n_sets)]
    for k in range(n_sets):
        groups = {}
        for i in range(n_sets):
            if i == k:
                continue
            v = inter_mask[i][k]
            if v not in groups:
                groups[v] = []
            groups[v].append(i)
        for S, group in groups.items():
            if len(group) < 2:
                continue
            for idx_a, idx_b in combinations(group, 2):
                a, b = idx_a, idx_b
                if inter_mask[a][b] == S:
                    pair = (min(a,b), max(a,b))
                    conflict_pairs[k].add(pair)
    
    def greedy_order(order):
        chosen = []
        chosen_set = set()
        chosen_pairs = set()
        for idx in order:
            if idx in chosen_set:
                continue
            has_conflict = False
            for pair in conflict_pairs[idx]:
                if pair in chosen_pairs:
                    has_conflict = True
                    break
            if not has_conflict:
                chosen.append(idx)
                chosen_set.add(idx)
                for c in chosen:
                    if c == idx:
                        continue
                    pair = (min(idx, c), max(idx, c))
                    chosen_pairs.add(pair)
        return chosen
    
    natural_order = list(range(n_sets))
    result = greedy_order(natural_order)
    best = list(result)
    best_size = len(best)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        for trial in range(1000):
            if time.time() > deadline - 10:
                break
            random.seed(trial * 53 + 37)
            order = list(range(n_sets))
            random.shuffle(order)
            res = greedy_order(order)
            if len(res) > best_size:
                best_size = len(res)
                best = res
    
    return [sorted(mask_to_set[masks[i]]) for i in best]
