import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
    forbidden_configs = []
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        
        for pairs in pairings:
            (a1,b1), (a2,b2), (a3,b3) = pairs
            transversals = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        t = tuple(sorted([(a1,b1)[i], (a2,b2)[j], (a3,b3)[k]]))
                        if t in triple_index:
                            transversals.append(triple_index[t])
            
            if len(transversals) == 8:
                forbidden_configs.append(frozenset(transversals))
    
    forbidden_configs = list(set(forbidden_configs))
    num_configs = len(forbidden_configs)
    
    config_to_triples = [list(c) for c in forbidden_configs]
    
    # Build triple_to_config_mask
    # Python int as bitset
    triple_to_config_mask = [0] * num_triples
    for ci, config in enumerate(forbidden_configs):
        mask_bit = 1 << ci
        for t in config:
            triple_to_config_mask[t] |= mask_bit

    # Greedy Hitting Set to get initial bound
    def greedy_hitting_set():
        active_mask = (1 << num_configs) - 1
        removed = set()
        
        while active_mask != 0:
            if time.time() > deadline - 5:
                break
            
            # Find a config
            # Pick the config with the most "unremoved" triples? 
            # Or just the first one.
            # Let's pick the config that has the most triples NOT in removed?
            # Actually, standard greedy: pick a set that covers the most uncovered elements.
            # Here, elements are configs. We pick a triple that covers the most active configs.
            
            # Count active configs per triple
            # This is expensive if done from scratch.
            # But num_triples=560.
            
            best_t = -1
            max_count = 0
            for t in range(num_triples):
                if t in removed:
                    continue
                # Count how many active configs contain t
                count = bin(triple_to_config_mask[t] & active_mask).count('1')
                if count > max_count:
                    max_count = count
                    best_t = t
            
            if best_t == -1:
                break
                
            removed.add(best_t)
            active_mask &= ~triple_to_config_mask[best_t]
            
        return removed

    initial_removed = greedy_hitting_set()
    best_removed_count = len(initial_removed)
    best_removed_set = initial_removed.copy()
    
    # B&B Solver
    # Global variables for B&B
    global_best_count = best_removed_count
    global_best_set = best_removed_set.copy()
    
    def greedy_packing_lb(mask):
        # Lower bound: Max number of disjoint active configs
        count = 0
        m = mask
        while m != 0:
            # Pick a config
            # To make it a valid packing, we must remove all intersecting configs.
            # Pick the lowest set bit
            c_idx = (m & -m).bit_length() - 1
            # Get the mask of all configs intersecting config c_idx
            # This requires precomputation or on-the-fly intersection.
            # Precompute config_intersects_mask?
            # config_intersects_mask[ci] is the mask of all configs that share at least one triple with config ci.
            # This is expensive to compute for 100k configs.
            
            # Alternative LB: 
            # Sum of (1 / max_cover) ? No.
            # Simple LB: 1 if mask != 0.
            # Better LB: Greedy packing.
            # To speed up greedy packing, we can just pick a config and subtract its "neighborhood".
            # We need `config_neighborhood_mask[ci]`.
            
            # Let's compute `config_neighborhood_mask` outside.
            # It might be memory heavy but 100k ints is fine.
            
            m &= ~config_neighborhood_mask[c_idx]
            count += 1
        return count

    # Precompute config_neighborhood_mask
    # For each config, which other configs intersect it?
    # Two configs intersect if they share a triple.
    # We can build this by iterating triples.
    # triple_to_configs is a list of lists.
    # For each triple t, let C_t be the set of configs containing t.
    # For each c1 in C_t, for each c2 in C_t, c1 intersects c2.
    
    config_neighborhood_mask = [0] * num_configs
    # We need triple_to_configs again
    triple_to_configs = [[] for _ in range(num_triples)]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
            
    for t in range(num_triples):
        configs_t = triple_to_configs[t]
        for c1 in configs_t:
            for c2 in configs_t:
                config_neighborhood_mask[c1] |= (1 << c2)

    def bb_solve(active_mask, removed_count, removed_set):
        nonlocal global_best_count, global_best_set
        
        if time.time() > deadline - 2:
            return
            
        if active_mask == 0:
            if removed_count < global_best_count:
                global_best_count = removed_count
                global_best_set = removed_set.copy()
            return
            
        # Bound
        lb = removed_count + greedy_packing_lb(active_mask)
        if lb >= global_best_count:
            return
            
        # Select a config to branch on
        # Heuristic: Pick a config with high degree? 
        # Or just the first one.
        c_idx = (active_mask & -active_mask).bit_length() - 1
        
        # Branch on triples in this config
        # Sort triples by number of active configs they hit (descending)
        candidates = config_to_triples[c_idx]
        
        # Filter out already removed? 
        # The recursion passes removed_set, but we branch on ALL triples in the config.
        # If a triple is already removed, it would have deactivated this config.
        # So all triples in c_idx are NOT in removed_set.
        
        def get_hit_count(t):
            return bin(triple_to_config_mask[t] & active_mask).count('1')
            
        candidates.sort(key=get_hit_count, reverse=True)
        
        for t in candidates:
            new_active_mask = active_mask & ~triple_to_config_mask[t]
            new_removed_set = removed_set | {t}
            bb_solve(new_active_mask, removed_count + 1, new_removed_set)
            
            if time.time() > deadline - 2:
                return

    # Run B&B
    # Initial active mask
    initial_active_mask = (1 << num_configs) - 1
    # We start with empty removed set? 
    # No, we want to improve upon the greedy solution.
    # So we can prune any branch that reaches removed_count >= global_best_count.
    # The greedy solution has size K.
    # B&B will try to find a solution of size < K.
    
    # Optimization: Start with the greedy removed set?
    # If we start with the greedy set, active_mask is 0, and we just confirm it.
    # We need to explore the space of removed sets.
    # Standard B&B starts with empty set and tries to build it.
    
    # To make it faster, we can use the greedy solution to guide the branching order?
    # Or just run B&B from scratch.
    
    # Given the time limit, B&B from scratch might not find a better solution than greedy if the search space is too large.
    # However, the bound is strong.
    
    # Let's try running B&B.
    bb_solve(initial_active_mask, 0, set())
    
    # If B&B didn't improve, use the greedy result.
    # If B&B improved, use the B&B result.
    
    final_removed = global_best_set
    
    # Construct the present list
    present = [True] * num_triples
    for t in final_removed:
        present[t] = False
        
    return [list(all_triples[i]) for i in range(num_triples) if present[i]]

