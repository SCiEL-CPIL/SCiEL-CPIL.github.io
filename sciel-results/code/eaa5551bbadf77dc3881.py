import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
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
    
    # Build incidence structure
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
    
    def can_add_triple(t, present, config_count):
        """Check if triple t can be added without violating any constraint."""
        for ci in triple_to_configs[t]:
            if config_count[ci] >= 7:
                return False
        return True
    
    def add_triple(t, present, config_count):
        """Add triple t to the system."""
        present[t] = True
        for ci in triple_to_configs[t]:
            config_count[ci] += 1
    
    def remove_triple(t, present, config_count):
        """Remove triple t from the system."""
        present[t] = False
        for ci in triple_to_configs[t]:
            config_count[ci] -= 1
    
    def greedy_construct(order=None):
        """Greedy construction with given ordering (or random if None)."""
        present = [False] * num_triples
        config_count = [0] * num_configs
        
        if order is None:
            order = list(range(num_triples))
            random.shuffle(order)
        
        for t in order:
            if time.time() > deadline - 5:
                break
            if can_add_triple(t, present, config_count):
                add_triple(t, present, config_count)
        
        return present, config_count
    
    def local_search(present, config_count, max_iters=200):
        """Local search with add and swap moves."""
        improved = True
        stall_count = 0
        
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try to add a triple - prioritize those with fewer constraints
            added = False
            for _ in range(100):
                t = random.randint(0, num_triples - 1)
                if present[t]:
                    continue
                if can_add_triple(t, present, config_count):
                    add_triple(t, present, config_count)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap: remove one triple, add another
            swapped = False
            for _ in range(100):
                t_remove = random.randint(0, num_triples - 1)
                if not present[t_remove]:
                    continue
                
                # Check which configs become available after removal
                slack_configs = set()
                for ci in triple_to_configs[t_remove]:
                    if config_count[ci] == 7:
                        slack_configs.add(ci)
                
                if not slack_configs:
                    continue
                
                # Try to find a triple that can be added
                t_add = random.randint(0, num_triples - 1)
                if present[t_add]:
                    continue
                
                # Check if t_add can be added after removing t_remove
                can_add = True
                for ci in triple_to_configs[t_add]:
                    if ci in slack_configs:
                        # After removing t_remove, this config has 6, so adding t_add makes 7 - OK
                        continue
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                
                if can_add:
                    # Perform the swap
                    remove_triple(t_remove, present, config_count)
                    add_triple(t_add, present, config_count)
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                stall_count = 0
                continue
            
            # No improvement found
            stall_count += 1
            if stall_count > 30:
                break
        
        return present, config_count
    
    def polish(present, config_count):
        """Final polish: systematically try all addable triples."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for t in range(num_triples):
                if present[t]:
                    continue
                if can_add_triple(t, present, config_count):
                    add_triple(t, present, config_count)
                    improved = True
        
        return present, config_count
    
    def annealing_search(present, config_count):
        """Simulated annealing style search."""
        temp = 1.0
        cooling = 0.99
        best_present = present[:]
        best_config_count = config_count[:]
        best_count = sum(present)
        
        while time.time() < deadline - 15 and temp > 0.1:
            # Random perturbation: remove a random present triple
            present_list = [i for i in range(num_triples) if present[i]]
            if not present_list:
                break
            
            t_remove = random.choice(present_list)
            remove_triple(t_remove, present, config_count)
            
            # Try to add some triples
            for _ in range(50):
                t = random.randint(0, num_triples - 1)
                if present[t]:
                    continue
                if can_add_triple(t, present, config_count):
                    add_triple(t, present, config_count)
            
            # Run local search to improve
            present, config_count = local_search(present, config_count, max_iters=100)
            
            current_count = sum(present)
            if current_count > best_count:
                best_count = current_count
                best_present = present[:]
                best_config_count = config_count[:]
            
            temp *= cooling
        
        return best_present, best_config_count
    
    best_present = None
    best_count = 0
    best_config_count = None
    
    # Deterministic baseline first
    det_order = list(range(num_triples))
    det_present, det_config_count = greedy_construct(det_order)
    det_present, det_config_count = local_search(det_present, det_config_count, max_iters=100)
    det_count = sum(det_present)
    
    if best_count < det_count:
        best_count = det_count
        best_present = det_present[:]
        best_config_count = det_config_count[:]
    
    # Run multiple randomized restarts
    num_restarts = 0
    while time.time() < deadline - 25:
        present, config_count = greedy_construct()
        present, config_count = local_search(present, config_count, max_iters=150)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present[:]
            best_config_count = config_count[:]
        
        num_restarts += 1
        
        # Occasionally try annealing from the best solution
        if num_restarts % 3 == 0 and best_present is not None and time.time() < deadline - 40:
            anneal_present, anneal_count = annealing_search(best_present[:], best_config_count[:])
            anneal_sum = sum(anneal_present)
            if anneal_sum > best_count:
                best_count = anneal_sum
                best_present = anneal_present[:]
                best_config_count = anneal_count[:]
        
        if num_restarts > 300:  # Safety cap
            break
    
    # Final polish on the best solution
    if best_present is not None:
        best_present, best_config_count = polish(best_present, best_config_count)
        best_count = sum(best_present)
    
    if best_present is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
