"""Triple systems on 16 points without K(2,2,2) - simulated annealing approach."""
import time
import random
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    total_triples = len(all_triples)
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []
    k222_config_index = {}
    
    for six_points in combinations(range(n), 6):
        a, b, c, d, e, f = six_points
        # 15 ways to partition 6 points into 3 unordered pairs
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        
        for p1, p2, p3 in partitions:
            x1, x2 = p1
            y1, y2 = p2
            z1, z2 = p3
            transversals = [
                (x1, y1, z1), (x1, y1, z2), (x1, y2, z1), (x1, y2, z2),
                (x2, y1, z1), (x2, y1, z2), (x2, y2, z1), (x2, y2, z2)
            ]
            transversals = [tuple(sorted(t)) for t in transversals]
            config_key = tuple(sorted(transversals))
            if config_key not in k222_config_index:
                k222_config_index[config_key] = len(k222_configs)
                k222_configs.append(config_key)
    
    num_configs = len(k222_configs)
    
    # For each triple, find which configs it belongs to
    triple_to_configs = [[] for _ in range(total_triples)]
    for config_id, config in enumerate(k222_configs):
        for t in config:
            triple_to_configs[triple_index[t]].append(config_id)
    
    deadline = time.time() + 115  # Safety margin
    
    # Initial solution: greedy add with random ordering
    random.seed(12345)
    selected = set()
    config_counts = [0] * num_configs
    
    triple_order = list(range(total_triples))
    random.shuffle(triple_order)
    
    for t_idx in triple_order:
        can_add = True
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] >= 7:
                can_add = False
                break
        if can_add:
            selected.add(t_idx)
            for cid in triple_to_configs[t_idx]:
                config_counts[cid] += 1
    
    best_selected = set(selected)
    best_size = len(selected)
    
    # Simulated annealing
    current_temp = 30.0
    cooling_rate = 0.9998
    min_temp = 0.05
    
    while time.time() < deadline and current_temp > min_temp:
        t_idx = random.randint(0, total_triples - 1)
        
        if t_idx in selected:
            # Try removing
            selected.remove(t_idx)
            for cid in triple_to_configs[t_idx]:
                config_counts[cid] -= 1
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)
        else:
            # Try adding
            can_add = True
            for cid in triple_to_configs[t_idx]:
                if config_counts[cid] >= 7:
                    can_add = False
                    break
            if can_add:
                selected.add(t_idx)
                for cid in triple_to_configs[t_idx]:
                    config_counts[cid] += 1
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
            else:
                # Can't add directly, try swap
                if len(selected) > 0 and random.random() < 0.5:
                    # Find a triple to remove that would allow adding t_idx
                    blocking_configs = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] >= 7]
                    if blocking_configs:
                        # Find candidates to remove from blocking configs
                        candidates = set()
                        for cid in blocking_configs:
                            for t2 in k222_configs[cid]:
                                t2_idx = triple_index[t2]
                                if t2_idx in selected and t2_idx != t_idx:
                                    candidates.add(t2_idx)
                        if candidates:
                            remove_idx = random.choice(list(candidates))
                            # Check if removing remove_idx allows adding t_idx
                            temp_counts = config_counts.copy()
                            for cid in triple_to_configs[remove_idx]:
                                temp_counts[cid] -= 1
                            can_swap = True
                            for cid in triple_to_configs[t_idx]:
                                if temp_counts[cid] >= 7:
                                    can_swap = False
                                    break
                            if can_swap:
                                selected.remove(remove_idx)
                                for cid in triple_to_configs[remove_idx]:
                                    config_counts[cid] -= 1
                                selected.add(t_idx)
                                for cid in triple_to_configs[t_idx]:
                                    config_counts[cid] += 1
                                if len(selected) > best_size:
                                    best_size = len(selected)
                                    best_selected = set(selected)
        
        current_temp *= cooling_rate
    
    # Final polish
    for t_idx in range(total_triples):
        if t_idx not in best_selected:
            can_add = True
            for cid in triple_to_configs[t_idx]:
                if config_counts[cid] >= 7:
                    can_add = False
                    break
            if can_add:
                best_selected.add(t_idx)
                for cid in triple_to_configs[t_idx]:
                    config_counts[cid] += 1
    
    return [list(all_triples[i]) for i in sorted(best_selected)]
