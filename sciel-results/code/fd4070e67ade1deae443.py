import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    # Map triple tuple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    for pts in itertools.combinations(points, 6):
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
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # Reverse map: triple_index -> list of config_ids
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    present = [False] * num_triples
    config_counts = [0] * num_configs
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] >= 7:
                return False
        return True

    def reset_state():
        for i in range(num_triples):
            present[i] = False
        for i in range(num_configs):
            config_counts[i] = 0

    def greedy_build(order):
        reset_state()
        current_set = []
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.append(t_idx)
        return current_set

    def local_search(current_set, max_iterations=200):
        """Improved local search with 1-swap and 2-swap moves."""
        # Load state
        reset_state()
        count = 0
        for idx in current_set:
            add_triple(idx)
            count += 1
        
        improved = True
        iter_count = 0
        no_improve_streak = 0
        while improved and iter_count < max_iterations and time.time() < deadline - 3:
            iter_count += 1
            improved = False
            
            # 1. Try to add any valid triple (greedy fill)
            missing = [i for i in range(num_triples) if not present[i]]
            random.shuffle(missing)
            
            added_any = False
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.append(t_idx)
                    count += 1
                    added_any = True
                    improved = True
                    
            if added_any:
                no_improve_streak = 0
                continue
            
            no_improve_streak += 1
            
            # 2. Try 1-swap: remove one present, add one missing
            if missing and current_set:
                # Sample a few missing triples
                sample_size = min(20, len(missing))
                candidates = random.sample(missing, sample_size)
                
                for t_idx in candidates:
                    if time.time() > deadline - 3:
                        break
                    conflicts = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] == 7]
                    if not conflicts:
                        continue
                    
                    # For each conflict, try removing one triple from that config
                    cid = random.choice(conflicts)
                    present_in_cid = [t for t in triples_in_config[cid] if present[t]]
                    if not present_in_cid:
                        continue
                    
                    random.shuffle(present_in_cid)
                    for u_idx in present_in_cid:
                        # Try: remove u_idx, add t_idx
                        remove_triple(u_idx)
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.remove(u_idx)
                            current_set.append(t_idx)
                            improved = True
                            no_improve_streak = 0
                            break
                        else:
                            add_triple(u_idx)
                        if improved:
                            break
                    if improved:
                        break
            
            # 3. Try 2-swap: remove two present, add one missing (net -1, but may unblock)
            if not improved and missing and len(current_set) >= 2:
                sample_size = min(10, len(missing))
                candidates = random.sample(missing, sample_size)
                
                for t_idx in candidates:
                    if time.time() > deadline - 3:
                        break
                    conflicts = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] == 7]
                    if not conflicts:
                        continue
                    
                    # Collect all triples that conflict with t_idx
                    conflict_triples = set()
                    for cid in conflicts:
                        for t in triples_in_config[cid]:
                            if present[t]:
                                conflict_triples.add(t)
                    
                    conflict_list = list(conflict_triples)
                    if len(conflict_list) < 2:
                        continue
                    
                    random.shuffle(conflict_list)
                    # Try pairs of removals
                    for i in range(min(5, len(conflict_list))):
                        for j in range(i+1, min(5, len(conflict_list))):
                            u1, u2 = conflict_list[i], conflict_list[j]
                            remove_triple(u1)
                            remove_triple(u2)
                            if is_valid_addition(t_idx):
                                add_triple(t_idx)
                                current_set.remove(u1)
                                current_set.remove(u2)
                                current_set.append(t_idx)
                                count -= 1
                                improved = True
                                no_improve_streak = 0
                                break
                            else:
                                add_triple(u1)
                                add_triple(u2)
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # 4. Try 2-for-1: remove two, add two different missing (net 0, reshuffle)
            if not improved and missing and len(current_set) >= 2:
                sample_missing = random.sample(missing, min(5, len(missing)))
                sample_present = random.sample(current_set, min(5, len(current_set)))
                
                for t_add1 in sample_missing:
                    for t_add2 in sample_missing:
                        if t_add1 == t_add2:
                            continue
                        for u_rem1 in sample_present:
                            for u_rem2 in sample_present:
                                if u_rem1 == u_rem2:
                                    continue
                                remove_triple(u_rem1)
                                remove_triple(u_rem2)
                                if is_valid_addition(t_add1) and is_valid_addition(t_add2):
                                    add_triple(t_add1)
                                    add_triple(t_add2)
                                    current_set.remove(u_rem1)
                                    current_set.remove(u_rem2)
                                    current_set.extend([t_add1, t_add2])
                                    improved = True
                                    no_improve_streak = 0
                                    break
                                else:
                                    add_triple(u_rem1)
                                    add_triple(u_rem2)
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved and no_improve_streak > 30:
                break
        
        return count, current_set

    best_set = []
    best_count = 0
    
    # Strategy 1: Random greedy + local search
    for seed in range(15):
        if time.time() > deadline - 5: break
        random.seed(seed)
        order = list(range(num_triples))
        random.shuffle(order)
        s = greedy_build(order)
        if len(s) > best_count:
            best_count = len(s)
            best_set = s[:]
        # Local search on this solution
        if len(s) >= best_count - 3:
            s_copy = s[:]
            count, s2 = local_search(s_copy, max_iterations=100)
            if count > best_count:
                best_count = count
                best_set = s2[:]
    
    # Strategy 2: Degree-based greedy (prefer triples with fewer conflicts)
    if time.time() < deadline - 10:
        # Compute conflict degree for each triple
        triple_degree = [len(triple_to_configs[i]) for i in range(num_triples)]
        for seed in range(5):
            if time.time() > deadline - 5: break
            random.seed(seed + 1000)
            # Sort by degree with some randomness
            order = list(range(num_triples))
            random.shuffle(order)
            order.sort(key=lambda x: (triple_degree[x], random.random()))
            s = greedy_build(order)
            if len(s) > best_count:
                best_count = len(s)
                best_set = s[:]
            if len(s) >= best_count - 3:
                s_copy = s[:]
                count, s2 = local_search(s_copy, max_iterations=150)
                if count > best_count:
                    best_count = count
                    best_set = s2[:]
    
    # Strategy 3: Deep local search on best solution
    if time.time() < deadline - 15 and best_set:
        for seed in range(10):
            if time.time() > deadline - 5: break
            random.seed(seed + 2000)
            s_copy = best_set[:]
            count, s = local_search(s_copy, max_iterations=300)
            if count > best_count:
                best_count = count
                best_set = s[:]
    
    # Strategy 4: More varied greedy + quick local search
    if time.time() < deadline - 10:
        for seed in range(50, 100):
            if time.time() > deadline - 5: break
            random.seed(seed)
            order = list(range(num_triples))
            random.shuffle(order)
            s = greedy_build(order)
            if len(s) > best_count:
                best_count = len(s)
                best_set = s[:]
            if len(s) >= best_count - 5:
                s_copy = s[:]
                count, s2 = local_search(s_copy, max_iterations=80)
                if count > best_count:
                    best_count = count
                    best_set = s2[:]
    
    # Final deep local search on the best
    if time.time() < deadline - 8 and best_set:
        s_copy = best_set[:]
        count, s = local_search(s_copy, max_iterations=500)
        if count > best_count:
            best_count = count
            best_set = s[:]

    result = [all_triples[i] for i in best_set]
    return result
