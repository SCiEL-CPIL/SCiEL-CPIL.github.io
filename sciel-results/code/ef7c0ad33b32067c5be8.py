import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
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
    
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
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
            if config_counts[cid] == 7:
                return False
        return True

    def get_initial_lines():
        # Affine lines in F2^4
        lines = []
        seen = set()
        for x in range(N):
            for y in range(N):
                if x == y: continue
                z = x ^ y
                if z == x or z == y: continue
                t = tuple(sorted((x, y, z)))
                if t not in seen:
                    seen.add(t)
                    lines.append(t)
        return lines

    def run_simulated_annealing(initial_set):
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        # Initialize with the provided set
        for t in initial_set:
            if is_valid_addition(triple_to_idx[t]):
                add_triple(triple_to_idx[t])
                current_set.add(triple_to_idx[t])
                
        current_count = len(current_set)
        best_count = current_count
        best_set = list(current_set)
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        max_steps = 50000
        
        random.seed(int(time.time() * 1000) % 10000)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
                
            # Generate a move: either add, remove, or 2-remove/1-add
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random missing triple
                missing = [i for i in range(num_triples) if not present[i]]
                if not missing:
                    continue
                t_idx = random.choice(missing)
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_set = list(current_set)
                else:
                    # Revert (no change)
                    pass
                    
            elif move_type < 0.5:
                # Try to remove a random present triple
                if not current_set:
                    continue
                t_idx = random.choice(list(current_set))
                remove_triple(t_idx)
                current_set.remove(t_idx)
                current_count -= 1
                # Accept with probability based on energy change (here, count change)
                # Delta E = -1 (we lost a triple)
                # Accept if exp(-delta_E/T) > random
                if random.random() > np.exp(-(-1) / T):
                    # Revert
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    current_count += 1
                    
            else:
                # 2-remove/1-add move
                if len(current_set) < 2:
                    continue
                to_remove = random.sample(list(current_set), 2)
                r1, r2 = to_remove
                
                # Find a triple that conflicts with r1 or r2 but is not present
                # This is a heuristic: pick a random missing triple
                missing = [i for i in range(num_triples) if not present[i]]
                if not missing:
                    continue
                t_idx = random.choice(missing)
                
                # Temporarily remove r1 and r2
                remove_triple(r1)
                remove_triple(r2)
                current_set.discard(r1)
                current_set.discard(r2)
                current_count -= 2
                
                # Try to add t_idx
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    current_count += 1
                    # Net change: -1
                    # Accept if exp(-(-1)/T) > random
                    if random.random() > np.exp(-(-1) / T):
                        # Revert: remove t_idx, add r1, r2
                        remove_triple(t_idx)
                        current_set.discard(t_idx)
                        current_count -= 1
                        add_triple(r1)
                        current_set.add(r1)
                        current_count += 1
                        add_triple(r2)
                        current_set.add(r2)
                        current_count += 1
                else:
                    # Revert: add r1, r2
                    add_triple(r1)
                    current_set.add(r1)
                    current_count += 1
                    add_triple(r2)
                    current_set.add(r2)
                    current_count += 1
                    
            if current_count > best_count:
                best_count = current_count
                best_set = list(current_set)
                
            T *= cooling_rate
            
        return best_count, best_set

    # Get the 35 affine lines
    initial_lines = get_initial_lines()
    
    best_count = 0
    best_set = []
    
    # Run simulated annealing starting from the algebraic structure
    count, s = run_simulated_annealing(initial_lines)
    if count > best_count:
        best_count = count
        best_set = s
        
    # If time remains, try a few random restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts
        # Random greedy start
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
        
        # Local search from random start
        init_triples = [all_triples[i] for i in current_set]
        count2, s2 = run_simulated_annealing(init_triples)
        
        if count2 > best_count:
            best_count = count2
            best_set = s2
            
        num_restarts += 1
        if num_restarts > 3: # Limit restarts to avoid overhead
            break

    result = [all_triples[i] for i in best_set]
    
    return result
