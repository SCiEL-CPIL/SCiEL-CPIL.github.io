import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices

    points = range(N)
    for pts in itertools.combinations(points, 6):
        # Generate all partitions of these 6 points into 3 pairs
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            # Normalize pair order and pair internal order for unique ID
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                # Find the 8 transversal triples
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # For each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    # Helper to add/remove
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False # Violation
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        # Check if any config it belongs to already has 7 others
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def get_missing():
        return [i for i in range(num_triples) if not present[i]]

    def run_walksat(seed_idx):
        random.seed(seed_idx)
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        # Initial random greedy fill
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
                
        count = len(current_set)
        
        # WalkSAT loop
        improved = True
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() < deadline - 2:
            # 1. Climb: Try to add any missing triple
            missing = get_missing()
            if not missing:
                break
                
            random.shuffle(missing)
            added_any = False
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    count += 1
                    added_any = True
                    no_improve_count = 0
                    if time.time() > deadline - 2:
                        break
            
            if not added_any:
                no_improve_count += 1
                
                # 2. Fluctuation: If stuck, remove a random triple to break a constraint
                if no_improve_count > max_no_improve or random.random() < 0.1:
                    # Pick a random present triple to remove
                    if current_set:
                        victim = random.choice(list(current_set))
                        remove_triple(victim)
                        current_set.remove(victim)
                        count -= 1
                        no_improve_count = 0
                        improved = True
                    else:
                        break
                else:
                    # Try a swap: pick a missing triple, remove one of its blockers, try to add
                    t_idx = random.choice(missing)
                    # Find a config blocking it
                    blockers = []
                    for cid in triple_to_configs[t_idx]:
                        if config_counts[cid] == 7:
                            blockers.append(cid)
                            
                    if blockers:
                        cid = random.choice(blockers)
                        # Pick a present triple in this config to remove
                        present_in_cid = [t for t in triples_in_config[cid] if present[t]]
                        if present_in_cid:
                            victim = random.choice(present_in_cid)
                            remove_triple(victim)
                            current_set.remove(victim)
                            count -= 1
                            
                            if is_valid_addition(t_idx):
                                add_triple(t_idx)
                                current_set.add(t_idx)
                                count += 1
                                no_improve_count = 0
                                improved = True
                            else:
                                # Failed swap, restore
                                add_triple(victim)
                                current_set.add(victim)
                                count += 1
                    else:
                        # No specific blocker? Shouldn't happen if missing and not addable, but safety
                        if current_set:
                            victim = random.choice(list(current_set))
                            remove_triple(victim)
                            current_set.remove(victim)
                            count -= 1
                            no_improve_count = 0
                        else:
                            break

        return count, list(current_set)

    best_count = 0
    best_set = []
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        seed = num_restarts
        count, s = run_walksat(seed)
        if count > best_count:
            best_count = count
            best_set = s
        num_restarts += 1
        
    # Convert indices back to triples
    result = [all_triples[i] for i in best_set]
    
    return result
