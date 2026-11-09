import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
    forbidden_configs = set()
    
    # Optimization: Iterate over 6-subsets
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        # There are 15 ways to partition 6 items into 3 pairs
        # We can generate these more efficiently or just list them as before
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
            # Unroll the loop for speed
            t0 = tuple(sorted([(a1,b1)[0], (a2,b2)[0], (a3,b3)[0]]))
            t1 = tuple(sorted([(a1,b1)[0], (a2,b2)[0], (a3,b3)[1]]))
            t2 = tuple(sorted([(a1,b1)[0], (a2,b2)[1], (a3,b3)[0]]))
            t3 = tuple(sorted([(a1,b1)[0], (a2,b2)[1], (a3,b3)[1]]))
            t4 = tuple(sorted([(a1,b1)[1], (a2,b2)[0], (a3,b3)[0]]))
            t5 = tuple(sorted([(a1,b1)[1], (a2,b2)[0], (a3,b3)[1]]))
            t6 = tuple(sorted([(a1,b1)[1], (a2,b2)[1], (a3,b3)[0]]))
            t7 = tuple(sorted([(a1,b1)[1], (a2,b2)[1], (a3,b3)[1]]))
            
            # Check if all exist (they always do since they are subsets of 6 distinct points)
            # We just need their indices
            idxs = []
            for t in (t0, t1, t2, t3, t4, t5, t6, t7):
                idxs.append(triple_index[t])
            
            forbidden_configs.add(frozenset(idxs))
            
    forbidden_configs = list(forbidden_configs)
    num_configs = len(forbidden_configs)
    
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
            
    # Precompute lengths for faster access? Not really needed.
    
    def greedy_delete():
        """Constraint-count-based greedy: prioritize triples in most active configs."""
        present = [True] * num_triples
        active = [True] * num_configs
        
        # Count how many active configs each triple participates in
        triple_active_count = [0] * num_triples
        for t in range(num_triples):
            for ci in triple_to_configs[t]:
                triple_active_count[t] += 1
        
        num_active = num_configs
        
        # Use a list of (count, triple) for faster max finding? 
        # Or just scan. 560 is small.
        
        while num_active > 0:
            if time.time() > deadline - 10:
                break
            
            # Find triple with max active count
            max_count = -1
            best_t = -1
            # Optimization: Only consider present triples
            for t in range(num_triples):
                if present[t]:
                    c = triple_active_count[t]
                    if c > max_count:
                        max_count = c
                        best_t = t
                        # Early break if max possible is 1? No, counts are higher.
                        
            if best_t == -1:
                break
                
            present[best_t] = False
            
            # Update active configs and counts
            for ci in triple_to_configs[best_t]:
                if active[ci]:
                    active[ci] = False
                    num_active -= 1
                    
                    for t_other in config_to_triples[ci]:
                        if t_other != best_t and present[t_other]:
                            triple_active_count[t_other] -= 1
                            
        return present
    
    def local_search_add(present):
        """Add triples that don't complete any forbidden config."""
        # Count how many triples from each config are currently present
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        # A triple t can be added if for all ci in triple_to_configs[t], config_count[ci] < 7
        # (Because adding t makes it 8)
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Find a safe triple to add
            # We can iterate through absent triples
            for t in range(num_triples):
                if not present[t]:
                    safe = True
                    for ci in triple_to_configs[t]:
                        if config_count[ci] >= 7:
                            safe = False
                            break
                    if safe:
                        present[t] = True
                        improved = True
                        # Update config counts
                        for ci in triple_to_configs[t]:
                            config_count[ci] += 1
                        break # Restart scan? Or continue? 
                        # Continuing is fine, but we might miss better ones. 
                        # However, any safe one is good. 
                        # To be more thorough, we could collect all safe ones and pick one.
                        # But for speed, greedy add is okay.
        
        return present

    def swap_search(present):
        """1-swap: remove one, add one. Try to improve density."""
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        # Precompute "danger" for absent triples: 
        # danger[t] = number of configs ci containing t where config_count[ci] >= 7
        # If danger[t] == 0, t is safe to add (if we don't remove anything).
        # If we remove r, danger[t] decreases for configs containing both t and r.
        
        # This is getting complex. Let's do a simple random swap.
        # Pick a random present triple r, remove it.
        # Pick a random absent triple t, try to add it.
        # If it fits, keep it.
        
        present_list = [t for t in range(num_triples) if present[t]]
        absent_list = [t for t in range(num_triples) if not present[t]]
        
        if not present_list or not absent_list:
            return present
            
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            r = random.choice(present_list)
            t = random.choice(absent_list)
            
            # Check if t is safe after removing r
            # t is in some configs. r is in some configs.
            # Intersection of configs(t) and configs(r) are the only ones affected.
            
            safe = True
            for ci in triple_to_configs[t]:
                # Current count in ci is config_count[ci]
                # If r is in ci, new count is config_count[ci] - 1
                # Else new count is config_count[ci]
                # We need new count + 1 (for t) <= 7 ? No, we need new count < 8.
                # i.e., new count <= 7.
                # Since t is being added, we check if (count_after_removal) + 1 <= 7?
                # No, the condition for validity is that NO config has 8 present triples.
                # So if we add t, any config containing t must have < 8 present triples.
                # Let C_t be configs containing t.
                # For ci in C_t:
                #   count_present_in_ci_after_change < 8
                #   count_present_in_ci_after_change = config_count[ci] - (1 if r in ci else 0) + 1
                #   We need config_count[ci] - (1 if r in ci else 0) + 1 < 8
                #   i.e., config_count[ci] - (1 if r in ci else 0) <= 7
                
                is_r_in_ci = False
                # Check if r is in config ci
                # This is O(1) if we had a set, but we have a list.
                # We can precompute config membership? 
                # Or just check: since ci contains t, and r is a triple, 
                # we can just check if r is in config_to_triples[ci]
                # This is a list of 8 ints. 
                
                # Optimization: Precompute a bitmask or set for each config? 
                # Or just rely on the fact that 8 is small.
                
                # Actually, we can check:
                # if r in config_to_triples[ci]:
                #    new_count = config_count[ci] - 1
                # else:
                #    new_count = config_count[ci]
                
                # If new_count >= 8, it's unsafe?
                # Wait. If new_count (before adding t) is 7, then adding t makes it 8. Unsafe.
                # If new_count is 6, adding t makes it 7. Safe.
                # So we need new_count <= 6?
                # Let's re-read: "no three disjoint pairs have all eight transversal triples present".
                # So a config is "broken" if it has <= 7 present triples.
                # So we need count_after_adding <= 7.
                # count_after_adding = config_count[ci] - (1 if r in ci else 0) + 1
                # We need config_count[ci] - (1 if r in ci else 0) + 1 <= 7
                # config_count[ci] - (1 if r in ci else 0) <= 6
                
                if r in config_to_triples[ci]:
                    if config_count[ci] - 1 > 6: # i.e., >= 7
                        safe = False
                        break
                else:
                    if config_count[ci] > 6: # i.e., >= 7
                        safe = False
                        break
                        
            if safe:
                # Perform swap
                present[r] = False
                present[t] = True
                
                # Update config counts
                for ci in triple_to_configs[r]:
                    config_count[ci] -= 1
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
                    
                # Update lists
                present_list.remove(r)
                present_list.append(t)
                absent_list.remove(t)
                absent_list.append(r)
                
                # After a successful swap, we might be able to add more?
                # local_search_add will handle that.
                return present # Or continue? Let's return to let outer loop handle restarts/adds.
        
        return present

    best_present = None
    best_count = 0
    
    # Run multiple instances
    num_restarts = 0
    while time.time() < deadline - 15:
        present = greedy_delete()
        present = local_search_add(present)
        present = swap_search(present)
        present = local_search_add(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present[:]
            
        num_restarts += 1
        if num_restarts > 50: # Limit restarts to ensure time for precomp if it was slow? 
            # Precomp is done once. 
            break
    
    if best_present is None:
        return []
        
    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
