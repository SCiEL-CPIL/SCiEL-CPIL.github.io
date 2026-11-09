import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)

    # Precompute all K(2,2,2) configurations (sets of 8 transversal triples)
    k222_configs = []
    for six in itertools.combinations(range(n), 6):
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
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            k222_configs.append(frozenset(transversal))

    num_configs = len(k222_configs)

    # For each config, store the 8 triple indices
    config_triple_indices = []
    for cfg in k222_configs:
        config_triple_indices.append([triple_index[t] for t in cfg])

    # For each triple, which configs it participates in
    triple_configs = [[] for _ in range(num_triples)]
    for ci, cfg_indices in enumerate(config_triple_indices):
        for ti in cfg_indices:
            triple_configs[ti].append(ci)

    # Greedy with constraint-count ordering
    def greedy_constrained(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        # Start with all triples available
        available = set(range(num_triples))
        selected = set()
        
        # Build list of (constraint_count, triple_idx)
        # constraint_count = number of configs this triple is in
        # But we want to pick triples that are "least constrained"
        # i.e., appear in fewer configs, so they block fewer future choices
        
        # Actually, for maximum independent set in hypergraph:
        # We want to greedily add triples. A triple can be added if it doesn't
        # complete any config. The "cost" of adding a triple is how many configs
        # it partially satisfies.
        
        # Simpler approach: random order but weighted by inverse constraint count
        # Let's try: pick triple with fewest configs first (less restrictive)
        
        # Precompute constraint counts
        constraint_counts = [len(triple_configs[i]) for i in range(num_triples)]
        
        # Shuffle with bias: lower constraint count = higher priority
        # Use weighted random selection
        items = list(range(num_triples))
        rng.shuffle(items)
        
        for ti in items:
            if ti not in available:
                continue
            # Check if we can add this triple
            can_add = True
            for ci in triple_configs[ti]:
                cfg_idx = config_triple_indices[ci]
                # Check if all other 7 are already selected
                all_present = True
                for other_ti in cfg_idx:
                    if other_ti == ti:
                        continue
                    if other_ti not in selected:
                        all_present = False
                        break
                if all_present:
                    can_add = False
                    break
            if can_add:
                selected.add(ti)
                available.discard(ti)
                # Remove all triples that would now be blocked
                # Actually, we don't need to remove them - we just check when adding
                # But for efficiency, we could track which triples are now "dangerous"
        
        return selected

    def greedy_random_order(seed_offset=0):
        rng = random.Random(1000 + seed_offset)
        items = list(range(num_triples))
        rng.shuffle(items)
        selected = set()
        for ti in items:
            can_add = True
            for ci in triple_configs[ti]:
                cfg_idx = config_triple_indices[ci]
                all_present = True
                for other_ti in cfg_idx:
                    if other_ti == ti:
                        continue
                    if other_ti not in selected:
                        all_present = False
                        break
                if all_present:
                    can_add = False
                    break
            if can_add:
                selected.add(ti)
        return selected

    def is_valid_selected(selected):
        """Check if selected set of triple indices is valid (no complete K(2,2,2))."""
        for ci in range(num_configs):
            cfg_idx = config_triple_indices[ci]
            if all(ti in selected for ti in cfg_idx):
                return False
        return True

    def local_search(selected, max_steps=5000, seed_offset=0):
        """2-swap local search: try removing one and adding two, or swapping one for one."""
        rng = random.Random(2000 + seed_offset)
        current = set(selected)
        current_list = list(current)
        
        # Precompute for faster checking
        # For each triple not in current, how many configs would it complete if added?
        # We'll use a violation counter approach
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try adding a random triple
                ti = rng.randrange(num_triples)
                if ti not in current:
                    can_add = True
                    for ci in triple_configs[ti]:
                        cfg_idx = config_triple_indices[ci]
                        all_present = True
                        for other_ti in cfg_idx:
                            if other_ti == ti:
                                continue
                            if other_ti not in current:
                                all_present = False
                                break
                        if all_present:
                            can_add = False
                            break
                    if can_add:
                        current.add(ti)
                        continue
            elif move_type < 0.7:
                # Try 1-swap: remove one, add one
                if current:
                    remove_ti = rng.choice(list(current))
                    current.discard(remove_ti)
                    found = False
                    for _ in range(20):
                        add_ti = rng.randrange(num_triples)
                        if add_ti != remove_ti and add_ti not in current:
                            can_add = True
                            for ci in triple_configs[add_ti]:
                                cfg_idx = config_triple_indices[ci]
                                all_present = True
                                for other_ti in cfg_idx:
                                    if other_ti == add_ti:
                                        continue
                                    if other_ti not in current:
                                        all_present = False
                                        break
                                if all_present:
                                    can_add = False
                                    break
                            if can_add:
                                current.add(add_ti)
                                found = True
                                break
                    if not found:
                        current.add(remove_ti)
            else:
                # Try 2-swap: remove two, add two (or remove one add two)
                if len(current) >= 2:
                    rem_list = list(current)
                    r1 = rng.choice(rem_list)
                    r2 = rng.choice([x for x in rem_list if x != r1])
                    current.discard(r1)
                    current.discard(r2)
                    
                    found = False
                    for _ in range(15):
                        a1 = rng.randrange(num_triples)
                        a2 = rng.randrange(num_triples)
                        if a1 == a2 or a1 in current or a2 in current:
                            continue
                        if a1 == r1 or a1 == r2 or a2 == r1 or a2 == r2:
                            continue
                        
                        # Check if both can be added
                        can_add_a1 = True
                        for ci in triple_configs[a1]:
                            cfg_idx = config_triple_indices[ci]
                            all_present = True
                            for other_ti in cfg_idx:
                                if other_ti == a1:
                                    continue
                                if other_ti not in current:
                                    all_present = False
                                    break
                            if all_present:
                                can_add_a1 = False
                                break
                        
                        if can_add_a1:
                            can_add_a2 = True
                            temp = set(current)
                            temp.add(a1)
                            for ci in triple_configs[a2]:
                                cfg_idx = config_triple_indices[ci]
                                all_present = True
                                for other_ti in cfg_idx:
                                    if other_ti == a2:
                                        continue
                                    if other_ti not in temp:
                                        all_present = False
                                        break
                                if all_present:
                                    can_add_a2 = False
                                    break
                            if can_add_a2:
                                current.add(a1)
                                current.add(a2)
                                found = True
                                break
                    
                    if not found:
                        current.add(r1)
                        current.add(r2)
        
        return current

    # Main search loop
    best_selected = set()
    best_size = 0

    random.seed(42)

    # Phase 1: Multiple greedy constructions
    num_greedy_rounds = 20
    for i in range(num_greedy_rounds):
        if time.time() > deadline - 30:
            break
        
        if i % 3 == 0:
            g_result = greedy_constrained(seed_offset=i)
        elif i % 3 == 1:
            g_result = greedy_random_order(seed_offset=i)
        else:
            g_result = greedy_random_order(seed_offset=100 + i)
        
        if len(g_result) > best_size:
            best_selected = set(g_result)
            best_size = len(g_result)
    
    # Phase 2: Local search on best and some random restarts
    if time.time() < deadline - 20:
        ls_result = local_search(best_selected, max_steps=3000, seed_offset=0)
        if is_valid_selected(ls_result):
            if len(ls_result) > best_size:
                best_selected = ls_result
                best_size = len(ls_result)

    # Phase 3: More local search with different seeds
    while time.time() < deadline - 10:
        # Start from best or a random greedy
        if time.time() < deadline - 30 and random.random() < 0.5:
            start = greedy_random_order(seed_offset=random.randint(0, 10000))
        else:
            start = set(best_selected)
        
        ls_result = local_search(start, max_steps=2000, seed_offset=random.randint(0, 10000))
        
        if is_valid_selected(ls_result):
            if len(ls_result) > best_size:
                best_selected = ls_result
                best_size = len(ls_result)
    
    # Final local search on best
    if time.time() < deadline - 3:
        final = local_search(best_selected, max_steps=2000, seed_offset=99999)
        if is_valid_selected(final):
            if len(final) > best_size:
                best_selected = final
                best_size = len(final)

    # Final validity check
    if not is_valid_selected(best_selected):
        # Fallback: rebuild a small valid set
        best_selected = set()
        # Try to add triples one by one
        for ti in range(num_triples):
            can_add = True
            for ci in triple_configs[ti]:
                cfg_idx = config_triple_indices[ci]
                all_present = True
                for other_ti in cfg_idx:
                    if other_ti == ti:
                        continue
                    if other_ti not in best_selected:
                        all_present = False
                        break
                if all_present:
                    can_add = False
                    break
            if can_add:
                best_selected.add(ti)

    # Convert to list of lists
    result = [list(all_triples[ti]) for ti in sorted(best_selected)]
    return result
