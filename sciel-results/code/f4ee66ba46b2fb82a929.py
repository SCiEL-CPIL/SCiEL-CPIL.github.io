import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all K(2,2,2) configurations
    # A config is defined by 3 disjoint pairs (a1,a2), (b1,b2), (c1,c2)
    # The 8 triples are (ai, bj, ck)
    
    # To efficiently generate configs, we iterate over 6 distinct vertices and their matchings
    configs = []
    # Precompute matchings for 6 vertices 0..5
    matchings = []
    verts = [0, 1, 2, 3, 4, 5]
    # Perfect matchings on 6 vertices
    # 0-1, 2-3, 4-5
    # 0-1, 2-4, 3-5
    # 0-1, 2-5, 3-4
    # 0-2, 1-3, 4-5
    # 0-2, 1-4, 3-5
    # 0-2, 1-5, 3-4
    # 0-3, 1-2, 4-5
    # 0-3, 1-4, 2-5
    # 0-3, 1-5, 2-4
    # 0-4, 1-2, 3-5
    # 0-4, 1-3, 2-5
    # 0-4, 1-5, 2-3
    # 0-5, 1-2, 3-4
    # 0-5, 1-3, 2-4
    # 0-5, 1-4, 2-3
    matchings = [
        [(0,1), (2,3), (4,5)],
        [(0,1), (2,4), (3,5)],
        [(0,1), (2,5), (3,4)],
        [(0,2), (1,3), (4,5)],
        [(0,2), (1,4), (3,5)],
        [(0,2), (1,5), (3,4)],
        [(0,3), (1,2), (4,5)],
        [(0,3), (1,4), (2,5)],
        [(0,3), (1,5), (2,4)],
        [(0,4), (1,2), (3,5)],
        [(0,4), (1,3), (2,5)],
        [(0,4), (1,5), (2,3)],
        [(0,5), (1,2), (3,4)],
        [(0,5), (1,3), (2,4)],
        [(0,5), (1,4), (2,3)]
    ]
    
    for verts6 in combinations(range(n), 6):
        v = list(verts6)
        for m in matchings:
            p1 = (v[m[0][0]], v[m[0][1]])
            p2 = (v[m[1][0]], v[m[1][1]])
            p3 = (v[m[2][0]], v[m[2][1]])
            # Generate the 8 triples
            t_set = set()
            for a in p1:
                for b in p2:
                    for c in p3:
                        t = tuple(sorted((a, b, c)))
                        t_set.add(t)
            configs.append(list(t_set))
            
    num_configs = len(configs)
    
    # Map triple to list of config indices containing it
    triple_to_configs = {t: [] for t in all_triples}
    for i, cfg in enumerate(configs):
        for t in cfg:
            triple_to_configs[t].append(i)
            
    def solve_once(seed):
        random.seed(seed)
        # present: set of triple indices
        present = set(range(len(all_triples)))
        # config_counts: count of present triples in each config
        config_counts = [8] * num_configs
        # active_configs: set of config indices with count 8
        active_configs = set(range(num_configs))
        
        # We need to efficiently pick a config and a triple.
        # To improve, we can prioritize configs with more "conflicting" triples?
        # Or just random.
        
        while active_configs:
            if time.time() > deadline:
                break
                
            # Pick a random active config
            cfg_idx = random.choice(list(active_configs))
            cfg_triples = configs[cfg_idx]
            
            # Pick a triple to remove. 
            # Heuristic: pick the triple that appears in the most other active configs?
            # This might be expensive to compute dynamically.
            # Let's start with random, or maybe pick the one with highest degree in the conflict graph restricted to active configs?
            # For now, random is simple.
            
            # Optimization: if we have many active configs, picking one at random is fine.
            # But we should remove a triple that helps fix other configs.
            # Let's calculate the "impact" of each triple in this config.
            # Impact = number of active configs that contain this triple.
            
            best_t = None
            best_impact = -1
            for t in cfg_triples:
                t_idx = triple_to_idx[t]
                if t_idx not in present:
                    continue
                # Count how many active configs contain t
                impact = 0
                for c_idx in triple_to_configs[t]:
                    if c_idx in active_configs:
                        impact += 1
                if impact > best_impact:
                    best_impact = impact
                    best_t = t_idx
            
            if best_t is None:
                # Should not happen if cfg_idx is active
                continue
                
            # Remove best_t
            present.remove(best_t)
            
            # Update counts for all configs containing best_t
            t_tuple = all_triples[best_t]
            for c_idx in triple_to_configs[t_tuple]:
                config_counts[c_idx] -= 1
                if config_counts[c_idx] == 7 and c_idx in active_configs:
                    active_configs.remove(c_idx)
                    
        return [list(all_triples[i]) for i in present]

    best_result = None
    best_count = -1
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for final return
            break
        result = solve_once(i)
        count = len(result)
        if count > best_count:
            best_count = count
            best_result = result
            
    if best_result is None:
        # Fallback to seed solution
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    return best_result
