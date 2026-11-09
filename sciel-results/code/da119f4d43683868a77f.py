import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all 6-point subsets and their K(2,2,2) transversal configurations
    # For each 6-set, there are 15 ways to partition into 3 pairs, each giving 8 transversal triples
    six_list = list(itertools.combinations(range(n), 6))
    num_six = len(six_list)
    
    # For each 6-set, store the 15 configurations, each being a list of 8 triple indices
    six_configs = []
    for six in six_list:
        p0, p1, p2, p3, p4, p5 = six
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
        configs = []
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                triple_index[tuple(sorted((a,c,e)))],
                triple_index[tuple(sorted((a,c,f)))],
                triple_index[tuple(sorted((a,d,e)))],
                triple_index[tuple(sorted((a,d,f)))],
                triple_index[tuple(sorted((b,c,e)))],
                triple_index[tuple(sorted((b,c,f)))],
                triple_index[tuple(sorted((b,d,e)))],
                triple_index[tuple(sorted((b,d,f)))],
            ]
            configs.append(transversal)
        six_configs.append(configs)
    
    # For each triple, store list of (six_idx, config_idx) where it appears
    triple_to_constraints = [[] for _ in range(num_triples)]
    for six_idx, configs in enumerate(six_configs):
        for config_idx, transversal in enumerate(configs):
            for t_idx in transversal:
                triple_to_constraints[t_idx].append((six_idx, config_idx))
    
    # Use a set of triple indices for fast membership
    # We'll use a Python set for the current solution
    
    def is_valid(mask_set, candidate_idx):
        """Check if adding candidate triple to mask_set creates a violation."""
        if candidate_idx in mask_set:
            return False
        for six_idx, config_idx in triple_to_constraints[candidate_idx]:
            transversal = six_configs[six_idx][config_idx]
            # Check if all 8 transversal triples are in mask_set (or is candidate)
            for t_idx in transversal:
                if t_idx == candidate_idx:
                    continue
                if t_idx not in mask_set:
                    break
            else:
                # All 8 present (7 in mask_set + 1 candidate)
                return False
        return True
    
    def greedy_order(order_indices):
        mask_set = set()
        count = 0
        for t_idx in order_indices:
            if is_valid(mask_set, t_idx):
                mask_set.add(t_idx)
                count += 1
        return mask_set, count
    
    def local_search(mask_set, max_steps=50000):
        """Local search with add and swap moves."""
        current = set(mask_set)
        best = set(mask_set)
        best_size = len(best)
        current_size = best_size
        
        all_indices = list(range(num_triples))
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Try adding a random triple
                t_idx = random.choice(all_indices)
                if t_idx not in current:
                    if is_valid(current, t_idx):
                        current.add(t_idx)
                        current_size += 1
                        if current_size > best_size:
                            best = set(current)
                            best_size = current_size
                        continue
            
            elif move_type < 0.85:
                # Try swapping: remove one, add another
                if current_size > 0:
                    # Pick a random element to remove
                    remove_idx = random.choice(list(current))
                    current.discard(remove_idx)
                    current_size -= 1
                    
                    found = False
                    for _ in range(50):
                        add_idx = random.choice(all_indices)
                        if add_idx != remove_idx and add_idx not in current:
                            if is_valid(current, add_idx):
                                current.add(add_idx)
                                current_size += 1
                                found = True
                                break
                    
                    if found:
                        if current_size > best_size:
                            best = set(current)
                            best_size = current_size
                    else:
                        current.add(remove_idx)
                        current_size += 1
            
            else:
                # Try 2-swap: remove two, add two
                if current_size >= 2:
                    current_list = list(current)
                    r1_idx = current_list[random.randint(0, len(current_list)-1)]
                    current_list.remove(r1_idx)
                    r2_idx = current_list[random.randint(0, len(current_list)-1)]
                    
                    current.discard(r1_idx)
                    current.discard(r2_idx)
                    current_size -= 2
                    
                    found1 = False
                    found2 = False
                    a1_idx = -1
                    a2_idx = -1
                    
                    for _ in range(50):
                        a1_candidate = random.choice(all_indices)
                        if a1_candidate not in current and a1_candidate != r1_idx and a1_candidate != r2_idx:
                            if is_valid(current, a1_candidate):
                                current.add(a1_candidate)
                                found1 = True
                                a1_idx = a1_candidate
                                break
                    
                    if found1:
                        for _ in range(50):
                            a2_candidate = random.choice(all_indices)
                            if a2_candidate not in current and a2_candidate != r1_idx and a2_candidate != r2_idx and a2_candidate != a1_idx:
                                if is_valid(current, a2_candidate):
                                    current.add(a2_candidate)
                                    found2 = True
                                    a2_idx = a2_candidate
                                    break
                    
                    if found1 and found2:
                        current_size += 2
                        if current_size > best_size:
                            best = set(current)
                            best_size = current_size
                    else:
                        # Revert
                        if found1:
                            current.discard(a1_idx)
                        if found2:
                            current.discard(a2_idx)
                        current.add(r1_idx)
                        current.add(r2_idx)
                        current_size += 2
        
        return best, best_size
    
    # Run multiple strategies
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy with constraint-count ordering (fewest constraints first)
    constraint_count = [len(c) for c in triple_to_constraints]
    order1 = sorted(range(num_triples), key=lambda i: constraint_count[i])
    mask1, size1 = greedy_order(order1)
    if size1 > best_size:
        best_set = mask1
        best_size = size1
    
    # Strategy 2: Greedy with reverse constraint-count ordering
    order2 = sorted(range(num_triples), key=lambda i: -constraint_count[i])
    mask2, size2 = greedy_order(order2)
    if size2 > best_size:
        best_set = mask2
        best_size = size2
    
    # Strategy 3: Deterministic sorted order
    order3 = list(range(num_triples))
    mask3, size3 = greedy_order(order3)
    if size3 > best_size:
        best_set = mask3
        best_size = size3
    
    # Local search on best
    if time.time() < deadline:
        ls_set, ls_size = local_search(best_set, max_steps=30000)
        if ls_size > best_size:
            best_set = ls_set
            best_size = ls_size
    
    # Multiple random restarts
    random.seed(42)
    
    while time.time() < deadline - 15:
        # Random greedy
        random_order = list(range(num_triples))
        random.shuffle(random_order)
        g_set, g_size = greedy_order(random_order)
        
        # Local search
        ls_set, ls_size = local_search(g_set, max_steps=20000)
        
        if ls_size > best_size:
            best_set = ls_set
            best_size = ls_size
    
    # Final polish
    if time.time() < deadline:
        final_set, final_size = local_search(best_set, max_steps=30000)
        if final_size > best_size:
            best_set = final_set
            best_size = final_size
    
    # Convert to list of lists
    result = [all_triples[i] for i in best_set]
    return [list(t) for t in sorted(result)]

