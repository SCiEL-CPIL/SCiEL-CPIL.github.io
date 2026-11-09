import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Safety margin for 150s kill
    
    n = 16
    vertices = list(range(n))
    all_edges = list(itertools.combinations(vertices, 3))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 560
    
    # Precompute matchings of 6 vertices into 3 pairs
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3)),
    ]
    
    # Build all configurations (sets of 8 transversal triples for 3 disjoint pairs)
    configs = []
    for subset in itertools.combinations(vertices, 6):
        v = list(subset)
        for m in matchings:
            p1 = (v[m[0][0]], v[m[0][1]])
            p2 = (v[m[1][0]], v[m[1][1]])
            p3 = (v[m[2][0]], v[m[2][1]])
            edges_in_config = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        edge = tuple(sorted((x1, x2, x3)))
                        edges_in_config.append(edge_to_idx[edge])
            configs.append(tuple(edges_in_config))
    
    num_configs = len(configs)  # 120120
    
    # Build configs_in_edge: for each edge, list of config indices containing it
    configs_in_edge = [[] for _ in range(num_edges)]
    for c_idx, c in enumerate(configs):
        for e_idx in c:
            configs_in_edge[e_idx].append(c_idx)
    
    # Convert to numpy arrays for fast access
    configs_in_edge_np = [np.array(lst, dtype=np.int32) for lst in configs_in_edge]
    
    # Precompute for each edge, the count of configs it belongs to
    edge_config_count = np.array([len(lst) for lst in configs_in_edge], dtype=np.int32)
    
    def run_greedy_with_lookahead(seed, time_budget):
        """
        Greedy with 1-step lookahead: when considering an edge, estimate
        how many future edges will be blocked. Prefer edges that block fewer.
        """
        random.seed(seed)
        order = list(range(num_edges))
        random.shuffle(order)
        
        conf_count = np.zeros(num_configs, dtype=np.int32)
        current_edges = []
        edge_selected = np.zeros(num_edges, dtype=bool)
        
        # For efficiency, track which configs are saturated (count == 7)
        # and which edges are "blocked" (can't be added because they'd violate)
        
        for e_idx in order:
            if time.time() > start_time + time_budget:
                break
            
            c_indices = configs_in_edge_np[e_idx]
            if len(c_indices) == 0:
                # No configs contain this edge, always safe
                current_edges.append(e_idx)
                edge_selected[e_idx] = True
                continue
            
            max_count = np.max(conf_count[c_indices])
            if max_count >= 7:
                continue  # Can't add
            
            # Lookahead: how many other unselected edges would become blocked
            # if we add this edge? This is expensive, so do it only sometimes.
            # For now, just add if safe.
            
            current_edges.append(e_idx)
            edge_selected[e_idx] = True
            conf_count[c_indices] += 1
        
        return current_edges
    
    def run_local_search(seed, time_budget):
        """
        Start from a greedy solution, then try to add removed edges
        by removing conflicting edges first (swap moves).
        """
        random.seed(seed)
        
        # First, get a greedy starting point
        random.seed(seed)
        order = list(range(num_edges))
        random.shuffle(order)
        
        conf_count = np.zeros(num_configs, dtype=np.int32)
        current_edges = []
        edge_selected = np.zeros(num_edges, dtype=bool)
        removed_edges = []
        
        for e_idx in order:
            c_indices = configs_in_edge_np[e_idx]
            if len(c_indices) > 0 and np.max(conf_count[c_indices]) >= 7:
                removed_edges.append(e_idx)
                continue
            current_edges.append(e_idx)
            edge_selected[e_idx] = True
            if len(c_indices) > 0:
                conf_count[c_indices] += 1
        
        # Local search: try to add removed edges by removing one selected edge
        # that conflicts with it
        improved = True
        local_deadline = start_time + time_budget
        
        while improved and time.time() < local_deadline:
            improved = False
            random.shuffle(removed_edges)
            
            for r_idx in removed_edges:
                if time.time() > local_deadline:
                    break
                
                c_indices_r = configs_in_edge_np[r_idx]
                if len(c_indices_r) == 0:
                    # No conflict, just add it
                    current_edges.append(r_idx)
                    edge_selected[r_idx] = True
                    removed_edges.remove(r_idx)
                    conf_count[c_indices_r] += 1
                    improved = True
                    continue
                
                # Find which selected edges conflict with r_idx
                # An edge s conflicts with r if they share a config where conf_count == 7
                # and both are in that config.
                
                # More precisely: adding r_idx to current set would violate config c
                # iff conf_count[c] == 7 and r_idx is in config c.
                # To add r_idx, we need to remove at least one edge from each such config c.
                
                violating_configs = []
                for c in c_indices_r:
                    if conf_count[c] >= 7:
                        violating_configs.append(c)
                
                if not violating_configs:
                    current_edges.append(r_idx)
                    edge_selected[r_idx] = True
                    removed_edges.remove(r_idx)
                    conf_count[c_indices_r] += 1
                    improved = True
                    continue
                
                # Try to remove one selected edge that fixes all violating configs
                # Find selected edges that are in at least one violating config
                candidate_removals = set()
                for c in violating_configs:
                    # Get all edges in config c that are currently selected
                    # We need to know which edges are in each config
                    # Let's precompute this
                    pass
                
                # This is getting complex. Let's use a simpler approach:
                # Try removing each selected edge in the violating configs one at a time
                # and check if r_idx can be added.
                
                # Collect candidate edges to remove
                cand_edges = set()
                for c in violating_configs:
                    # We need edges in config c that are selected
                    # Precompute config_to_edges
                    pass
        
        return current_edges
    
    # Precompute config_to_edges for local search
    config_to_edges = [np.array(list(c), dtype=np.int32) for c in configs]
    
    def run_local_search_v2(seed, time_budget):
        """
        Start from greedy, then try swap moves: remove one edge, add another.
        """
        random.seed(seed)
        
        # Get greedy starting point
        order = list(range(num_edges))
        random.shuffle(order)
        
        conf_count = np.zeros(num_configs, dtype=np.int32)
        current_edges_set = set()
        edge_selected = np.zeros(num_edges, dtype=bool)
        
        for e_idx in order:
            c_indices = configs_in_edge_np[e_idx]
            if len(c_indices) > 0 and np.max(conf_count[c_indices]) >= 7:
                continue
            current_edges_set.add(e_idx)
            edge_selected[e_idx] = True
            if len(c_indices) > 0:
                conf_count[c_indices] += 1
        
        local_deadline = start_time + time_budget
        best_size = len(current_edges_set)
        no_improve_count = 0
        
        while time.time() < local_deadline and no_improve_count < 20:
            no_improve_count += 1
            
            # Try random swap moves
            num_attempts = 100
            for _ in range(num_attempts):
                if time.time() > local_deadline:
                    break
                
                # Pick a random unselected edge to try adding
                unselected = [i for i in range(num_edges) if not edge_selected[i]]
                if not unselected:
                    break
                
                add_e = random.choice(unselected)
                c_indices_add = configs_in_edge_np[add_e]
                
                # Check if adding is possible
                if len(c_indices_add) > 0 and np.max(conf_count[c_indices_add]) >= 7:
                    # Need to remove something
                    # Find a selected edge that's in a violating config
                    violating = [c for c in c_indices_add if conf_count[c] >= 7]
                    if not violating:
                        continue
                    
                    # Pick a random violating config
                    v_c = random.choice(violating)
                    # Pick a random selected edge in that config
                    edges_in_vc = config_to_edges[v_c]
                    selected_in_vc = [e for e in edges_in_vc if edge_selected[e]]
                    if not selected_in_vc:
                        continue
                    
                    rem_e = random.choice(selected_in_vc)
                    
                    # Check if removing rem_e and adding add_e is valid
                    # Simulate: decrement conf_count for rem_e's configs,
                    # check if add_e can be added, then increment.
                    
                    c_indices_rem = configs_in_edge_np[rem_e]
                    
                    # Check: after removing rem_e, can we add add_e?
                    # For each config in add_e's configs, count after removal:
                    # new_count[c] = conf_count[c] - (1 if rem_e in c else 0)
                    # We need new_count[c] < 7 for all c in add_e's configs.
                    
                    can_swap = True
                    for c in c_indices_add:
                        rem_in_c = 1 if c in set(c_indices_rem) else 0
                        # Wait, c_indices_rem is a numpy array, checking membership is slow
                        # Let's use a set for c_indices_rem
                        pass
                    
                    # Better: precompute a boolean matrix or use sets
                    # For now, let's just check directly
                    rem_set = set(c_indices_rem.tolist()) if len(c_indices_rem) > 0 else set()
                    
                    for c in c_indices_add:
                        new_count = conf_count[c] - (1 if c in rem_set else 0)
                        if new_count >= 7:
                            can_swap = False
                            break
                    
                    if not can_swap:
                        continue
                    
                    # Also check: after adding add_e, no config exceeds 7
                    # This is already checked above (new_count < 7)
                    
                    # Perform the swap
                    # Remove rem_e
                    current_edges_set.remove(rem_e)
                    edge_selected[rem_e] = False
                    for c in c_indices_rem:
                        conf_count[c] -= 1
                    
                    # Add add_e
                    current_edges_set.add(add_e)
                    edge_selected[add_e] = True
                    for c in c_indices_add:
                        conf_count[c] += 1
                    
                    if len(current_edges_set) > best_size:
                        best_size = len(current_edges_set)
                        no_improve_count = 0
                else:
                    # Can add directly
                    current_edges_set.add(add_e)
                    edge_selected[add_e] = True
                    for c in c_indices_add:
                        conf_count[c] += 1
                    if len(current_edges_set) > best_size:
                        best_size = len(current_edges_set)
                        no_improve_count = 0
        
        return list(current_edges_set)
    
    # Convert config_to_edges to sets for fast membership
    config_to_edges_set = [set(c.tolist()) for c in config_to_edges]
    
    best_system = []
    best_count = 0
    
    # Phase 1: Random greedy restarts
    greedy_deadline = start_time + 40
    num_runs = 0
    while time.time() < greedy_deadline:
        seed = num_runs
        result = run_greedy_with_lookahead(seed, 0)
        if len(result) > best_count:
            best_count = len(result)
            best_system = result
        num_runs += 1
    
    # Phase 2: Local search from best solution
    if time.time() < deadline - 5:
        ls_deadline = deadline - 3
        best_ls = run_local_search_v2(0, ls_deadline - start_time)
        if len(best_ls) > best_count:
            best_count = len(best_ls)
            best_system = best_ls
    
    return [list(all_edges[i]) for i in sorted(best_system)]
