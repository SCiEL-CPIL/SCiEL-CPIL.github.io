import time
import random
from itertools import combinations

def construct():
    # Deadline guard
    deadline = time.time() + 140
    
    n = 16
    points = list(range(n))
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute configurations
    # A configuration is a set of 8 transversal triples from 3 disjoint pairs.
    config_triples = []
    triple_configs = [[] for _ in range(num_triples)]
    
    # Helper to get transversal triple indices for 3 pairs
    def get_transversal_indices(pairs):
        a1, a2 = pairs[0]
        b1, b2 = pairs[1]
        c1, c2 = pairs[2]
        idxs = []
        for a in (a1, a2):
            for b in (b1, b2):
                for c in (c1, c2):
                    t = tuple(sorted((a, b, c)))
                    idxs.append(triple_to_idx[t])
        return idxs

    # Iterate over all sets of 6 points
    for six_pts in combinations(range(n), 6):
        p = list(six_pts)
        # Generate 15 perfect matchings
        for i in range(1, 6):
            pair1 = (p[0], p[i])
            remaining = [p[j] for j in range(6) if j != 0 and j != i]
            for k in range(1, 4):
                pair2 = (remaining[0], remaining[k])
                remaining2 = [remaining[j] for j in range(4) if j != 0 and j != k]
                pair3 = (remaining2[0], remaining2[1])
                
                idxs = get_transversal_indices([pair1, pair2, pair3])
                
                config_id = len(config_triples)
                config_triples.append(idxs)
                
                for idx in idxs:
                    triple_configs[idx].append(config_id)
    
    num_configs = len(config_triples)
    
    def run_greedy(order):
        selected = [False] * num_triples
        counts = [0] * num_configs
        count_selected = 0
        tc = triple_configs
        
        for t_idx in order:
            can_add = True
            for c_idx in tc[t_idx]:
                if counts[c_idx] == 7:
                    can_add = False
                    break
            
            if can_add:
                selected[t_idx] = True
                count_selected += 1
                for c_idx in tc[t_idx]:
                    counts[c_idx] += 1
        return count_selected, selected

    best_count = 0
    best_selected = None
    
    # Initial run
    order = list(range(num_triples))
    c, sel = run_greedy(order)
    if c > best_count:
        best_count = c
        best_selected = sel
        
    # Random restarts
    while time.time() < deadline:
        order = list(range(num_triples))
        random.shuffle(order)
        c, sel = run_greedy(order)
        if c > best_count:
            best_count = c
            best_selected = sel
            
    # Convert to output format
    result = []
    for i, is_sel in enumerate(best_selected):
        if is_sel:
            result.append(list(all_triples[i]))
            
    return result

random.seed(42)
