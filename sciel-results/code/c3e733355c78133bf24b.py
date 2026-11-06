import random
import time
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = idx
                edges.append((u, v))
                idx += 1
    
    n_edges = len(edges)
    
    # Precompute conflicts: for each edge e, list of triples (a,b,c) such that
    # if a,b,c are all in the set, e cannot be added
    conflicts = [[] for _ in range(n_edges)]
    
    for e_idx, (u, v) in enumerate(edges):
        diff = u ^ v
        bit_i = diff.bit_length() - 1
        
        for bit_j in range(d):
            if bit_j == bit_i:
                continue
            mask_j = 1 << bit_j
            u_prime = u ^ mask_j
            v_prime = v ^ mask_j
            
            e1 = edge_map.get((min(u, u_prime), max(u, u_prime)))
            e2 = edge_map.get((min(v, v_prime), max(v, v_prime)))
            e3 = edge_map.get((min(u_prime, v_prime), max(u_prime, v_prime)))
            
            if e1 is not None and e2 is not None and e3 is not None:
                conflicts[e_idx].append((e1, e2, e3))
    
    # For each edge x, precompute which (f, triple_idx, position) it participates in
    edge_in_triples = [[] for _ in range(n_edges)]
    for f in range(n_edges):
        for t_idx, triple in enumerate(conflicts[f]):
            for pos, edge in enumerate(triple):
                edge_in_triples[edge].append((f, t_idx, pos))
    
    in_set = [0] * n_edges
    blockers = [0] * n_edges
    current_count = 0
    
    def can_add(e_idx):
        return blockers[e_idx] == 0
    
    def add_edge(e_idx):
        nonlocal current_count
        if in_set[e_idx] or blockers[e_idx] > 0:
            return False
        in_set[e_idx] = 1
        current_count += 1
        for (f, t_idx, pos) in edge_in_triples[e_idx]:
            triple = conflicts[f][t_idx]
            o1 = triple[(pos + 1) % 3]
            o2 = triple[(pos + 2) % 3]
            if in_set[o1] and in_set[o2]:
                blockers[f] += 1
        return True
    
    def remove_edge(e_idx):
        nonlocal current_count
        if not in_set[e_idx]:
            return False
        in_set[e_idx] = 0
        current_count -= 1
        for (f, t_idx, pos) in edge_in_triples[e_idx]:
            triple = conflicts[f][t_idx]
            o1 = triple[(pos + 1) % 3]
            o2 = triple[(pos + 2) % 3]
            if in_set[o1] and in_set[o2]:
                blockers[f] -= 1
        return True
    
    def reset_state():
        nonlocal in_set, blockers, current_count
        in_set = [0] * n_edges
        blockers = [0] * n_edges
        current_count = 0
    
    def local_polish():
        """Iteratively try to add any addable edge until no more can be added."""
        nonlocal current_count
        improved = True
        while improved:
            improved = False
            for e_idx in range(n_edges):
                if not in_set[e_idx] and can_add(e_idx):
                    add_edge(e_idx)
                    improved = True
        return current_count
    
    def simulated_annealing(T_start, T_end, cooling, time_budget):
        """Run SA for a given time budget, return (best_count, best_set_copy)."""
        nonlocal in_set, blockers, current_count
        sa_best_count = current_count
        sa_best_set = in_set[:]
        T = T_start
        all_edges = list(range(n_edges))
        sa_start = time.time()
        
        while time.time() < sa_start + time_budget:
            r = random.random()
            if r < 0.35:
                e_idx = random.choice(all_edges)
                if not in_set[e_idx] and can_add(e_idx):
                    add_edge(e_idx)
                    if current_count > sa_best_count:
                        sa_best_count = current_count
                        sa_best_set = in_set[:]
            elif r < 0.55:
                if current_count > 0:
                    e_idx = random.choice(all_edges)
                    if in_set[e_idx]:
                        remove_edge(e_idx)
                        if random.random() > 2.718281828 ** (-1.0 / T):
                            add_edge(e_idx)
            else:
                if 0 < current_count < n_edges:
                    e_out = random.choice(all_edges)
                    if in_set[e_out]:
                        e_in = random.choice(all_edges)
                        if not in_set[e_in]:
                            remove_edge(e_out)
                            if can_add(e_in):
                                add_edge(e_in)
                            else:
                                add_edge(e_out)
            
            T *= cooling
            if T < T_end:
                T = T_end
        
        return sa_best_count, sa_best_set
    
    start_time = time.time()
    deadline = start_time + 115.0
    all_edges = list(range(n_edges))
    
    # Multi-restart strategy with varying initial conditions
    global_best_count = 0
    global_best_set = [0] * n_edges
    
    # Phase 1: Greedy init + SA
    for e_idx in range(n_edges):
        add_edge(e_idx)
    local_polish()
    
    best_count, best_set = simulated_annealing(
        T_start=8.0, T_end=0.01, cooling=0.99997, 
        time_budget=min(60.0, deadline - time.time())
    )
    if best_count > global_best_count:
        global_best_count = best_count
        global_best_set = best_set
    
    # Phase 2: Random restarts with different initializations
    num_restarts = 5
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        time_per_restart = remaining_time / (num_restarts + 1)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Different initialization strategies
            reset_state()
            
            if restart % 2 == 0:
                # Random order greedy
                shuffled = all_edges[:]
                random.shuffle(shuffled)
                for e_idx in shuffled:
                    add_edge(e_idx)
            else:
                # Random subset then polish
                for e_idx in all_edges:
                    if random.random() < 0.7:
                        add_edge(e_idx)
            
            local_polish()
            
            best_count, best_set = simulated_annealing(
                T_start=5.0, T_end=0.005, cooling=0.99995,
                time_budget=min(time_per_restart, deadline - time.time() - 1.0)
            )
            if best_count > global_best_count:
                global_best_count = best_count
                global_best_set = best_set
    
    # Final polish on the best solution found
    reset_state()
    for e_idx in range(n_edges):
        if global_best_set[e_idx]:
            in_set[e_idx] = 1
            current_count += 1
            for (f, t_idx, pos) in edge_in_triples[e_idx]:
                triple = conflicts[f][t_idx]
                o1 = triple[(pos + 1) % 3]
                o2 = triple[(pos + 2) % 3]
                if in_set[o1] and in_set[o2]:
                    blockers[f] += 1
    
    # Try to add more edges in the final polish
    for e_idx in range(n_edges):
        if not in_set[e_idx] and can_add(e_idx):
            add_edge(e_idx)
    
    # One more round of local search with small perturbations
    if time.time() < deadline - 2.0:
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            # Randomly remove a few edges and re-polish
            for _ in range(random.randint(3, 10)):
                e_idx = random.choice(all_edges)
                if in_set[e_idx]:
                    remove_edge(e_idx)
            local_polish()
            if current_count > global_best_count:
                global_best_count = current_count
                global_best_set = in_set[:]
    
    result = []
    for e_idx in range(n_edges):
        if global_best_set[e_idx]:
            result.append([edges[e_idx][0], edges[e_idx][1]])
    return result
