import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
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
    
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
    
    def greedy_delete():
        """Constraint-count-based greedy: prioritize triples in most active configs."""
        present = [True] * num_triples
        active = [True] * num_configs
        
        # Count how many active configs each triple participates in
        triple_active_count = [0] * num_triples
        for t in range(num_triples):
            for ci in triple_to_configs[t]:
                if active[ci]:
                    triple_active_count[t] += 1
        
        num_active = num_configs
        
        while num_active > 0:
            if time.time() > deadline - 10:
                break
            
            # Find triples with maximum active config count
            max_count = -1
            candidates = []
            for t in range(num_triples):
                if present[t]:
                    if triple_active_count[t] > max_count:
                        max_count = triple_active_count[t]
                        candidates = [t]
                    elif triple_active_count[t] == max_count:
                        candidates.append(t)
            
            if not candidates:
                break
            
            best_t = random.choice(candidates)
            
            present[best_t] = False
            
            for ci in triple_to_configs[best_t]:
                if active[ci]:
                    active[ci] = False
                    num_active -= 1
                    
                    for t_other in config_to_triples[ci]:
                        if t_other != best_t and present[t_other]:
                            triple_active_count[t_other] -= 1
        
        return present
    
    def local_search_add(present):
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            candidates = []
            for t in range(num_triples):
                if not present[t] and unsafe_count[t] == 0:
                    candidates.append(t)
            
            if candidates:
                t = random.choice(candidates)
                present[t] = True
                improved = True
                
                for ci in triple_to_configs[t]:
                    old_count = config_count[ci]
                    config_count[ci] = old_count + 1
                    
                    if old_count == 6:
                        for t_other in config_to_triples[ci]:
                            if t_other != t and not present[t_other]:
                                unsafe_count[t_other] += 1
        
        return present
    
    def swap_search(present):
        """Efficient 1-swap: remove one present triple, add one absent triple."""
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        # For each absent triple, precompute which configs would block it
        # unsafe[t] = number of configs where config_count >= 7
        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Find all safe absent triples
            safe_absent = [t for t in range(num_triples) if not present[t] and unsafe_count[t] == 0]
            
            if not safe_absent:
                break
            
            # For each safe absent triple, try to find a present triple to remove
            # that doesn't conflict with it (i.e., they don't share a config with count >= 7 after removal)
            # Simplified: just try random swaps
            
            present_list = [t for t in range(num_triples) if present[t]]
            
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                
                t_add = random.choice(safe_absent)
                t_remove = random.choice(present_list)
                
                # Check: after removing t_remove, is t_add still safe?
                # t_add is safe if for all configs ci containing t_add,
                # the new count (old count - 1 if t_remove in ci, else old count) < 8
                # i.e., old count < 8 if t_remove not in ci, or old count - 1 < 8 if t_remove in ci
                # Since t_add was already safe (unsafe_count[t_add] == 0), all counts for t_add's configs are <= 6
                # After removing t_remove, counts can only decrease, so t_add remains safe!
                # So any swap where t_add is currently safe is valid.
                
                # But we need the swap to be beneficial: we gain 1 triple (add t_add) and lose 1 (remove t_remove)
                # Net zero, but it might enable future additions.
                # Let's do it and then try to add more.
                
                present[t_remove] = False
                present[t_add] = True
                
                # Update config_count
                for ci in triple_to_configs[t_remove]:
                    config_count[ci] -= 1
                for ci in triple_to_configs[t_add]:
                    config_count[ci] += 1
                
                # Update unsafe_count for affected absent triples
                # Only triples in configs that changed need updating
                affected_configs = set(triple_to_configs[t_remove]) | set(triple_to_configs[t_add])
                for ci in affected_configs:
                    for t_other in config_to_triples[ci]:
                        if not present[t_other] and t_other != t_add:
                            # Recompute unsafe for t_other
                            # This is expensive, so let's just recompute all at the end
                            pass
                
                # Full recompute of unsafe_count (expensive but correct)
                unsafe_count = [0] * num_triples
                for t in range(num_triples):
                    if not present[t]:
                        for ci in triple_to_configs[t]:
                            if config_count[ci] >= 7:
                                unsafe_count[t] += 1
                
                improved = True
                break
        
        # After swaps, try to add more
        present = local_search_add(present)
        
        return present
    
    def remove_and_readd(present):
        """Try removing a present triple and re-adding different ones."""
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            present_list = [t for t in range(num_triples) if present[t]]
            
            if not present_list:
                break
            
            # Try removing each present triple and see if we can add more back
            for t_remove in random.sample(present_list, min(len(present_list), 20)):
                if time.time() > deadline - 5:
                    break
                
                present[t_remove] = False
                for ci in triple_to_configs[t_remove]:
                    config_count[ci] -= 1
                
                # Try to add back as many as possible
                count_before = sum(present)
                present = local_search_add(present)
                count_after = sum(present)
                
                if count_after > count_before:
                    # Good, keep this
                    improved = True
                    break
                else:
                    # Revert
                    present[t_remove] = True
                    for ci in triple_to_configs[t_remove]:
                        config_count[ci] += 1
        
        return present
    
    best_present = None
    best_count = 0
    
    # Run multiple instances
    num_restarts = 0
    while time.time() < deadline - 15:
        present = greedy_delete()
        present = local_search_add(present)
        present = swap_search(present)
        present = remove_and_readd(present)
        present = local_search_add(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present[:]
        
        num_restarts += 1
        if num_restarts > 200:
            break
    
    if best_present is None:
        return []
    
    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
