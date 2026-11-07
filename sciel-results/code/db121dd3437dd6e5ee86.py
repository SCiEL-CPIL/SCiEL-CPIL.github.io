import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def get_score(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_get_score(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods)), sums, prods

    def generate_regular_graph(degree, seed):
        """Generate a near-regular graph on n vertices with m edges."""
        rng = random.Random(seed)
        # Start with a random graph and try to make it near-regular
        best_graph = None
        best_var = float('inf')
        
        for attempt in range(50):
            edges = set()
            # Start with a cycle
            for i in range(n):
                edges.add((i, (i + 1) % n))
            
            # Add random edges until we have m
            while len(edges) < m:
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u != v:
                    e = (min(u, v), max(u, v))
                    edges.add(e)
            
            if len(edges) > m:
                # Remove excess edges
                edge_list = list(edges)
                rng.shuffle(edge_list)
                edges = set(edge_list[:m])
            
            # Compute degree variance
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            mean_deg = sum(deg) / n
            var = sum((d - mean_deg) ** 2 for d in deg) / n
            
            if var < best_var:
                best_var = var
                best_graph = edges
        
        return best_graph

    def generate_complete_subgraph_plus_random(seed):
        """Generate a graph with a dense core plus random spokes."""
        rng = random.Random(seed)
        edges = set()
        
        # Core: first 7 vertices form K7 (21 edges)
        for i in range(7):
            for j in range(i + 1, 7):
                edges.add((i, j))
        
        # Add random edges involving remaining vertices
        remaining = list(range(7, n))
        all_possible = [e for e in all_edges if e not in edges]
        rng.shuffle(all_possible)
        
        while len(edges) < m and all_possible:
            e = all_possible.pop()
            edges.add(e)
        
        return edges

    def generate_star_plus_clique(seed):
        """Generate a star-like structure with a small clique."""
        rng = random.Random(seed)
        edges = set()
        
        # Center vertex 0 connected to all others
        for i in range(1, n):
            edges.add((0, i))
        
        # Add edges among non-center vertices
        non_center = list(range(1, n))
        possible = [e for e in all_edges if e not in edges]
        rng.shuffle(possible)
        
        while len(edges) < m and possible:
            e = possible.pop()
            edges.add(e)
        
        return edges

    def collision_seeking_optimize(values, edges, max_iters=30000, target_k=35):
        """
        Optimize values by explicitly seeking collisions.
        Each move tries to make an edge's sum or product match an existing one.
        """
        current_values = list(values)
        edge_list = list(edges)
        current_score, current_sums, current_prods = fast_get_score(current_values, edge_list)
        
        best_score = current_score
        best_values = list(current_values)
        
        # Build adjacency for quick neighbor lookup
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        used_values = set(current_values)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to modify
            idx = random.randint(0, n - 1)
            neighbors = list(adj[idx])
            if not neighbors:
                continue
            
            old_val = current_values[idx]
            
            # Strategy 1: Try to match an existing sum
            # For each neighbor, find a value that makes sum match an existing sum
            candidates = []
            for nb in neighbors:
                v_nb = current_values[nb]
                # Try to match existing sums
                for s in current_sums:
                    cand = s - v_nb
                    if 1 <= cand <= 10000 and cand != old_val:
                        if cand not in used_values or cand == old_val:
                            candidates.append(cand)
                # Try to match existing products
                for p in current_prods:
                    if v_nb != 0 and p % v_nb == 0:
                        cand = p // v_nb
                        if 1 <= cand <= 10000 and cand != old_val:
                            if cand not in used_values or cand == old_val:
                                candidates.append(cand)
            
            # Strategy 2: Try to create a new collision by matching two edges
            # Pick two neighbors and try to make their sums equal
            if len(neighbors) >= 2:
                nb1, nb2 = random.sample(neighbors, 2)
                v1 = current_values[nb1]
                v2 = current_values[nb2]
                # We want val + v1 == val + v2, which is impossible unless v1==v2
                # Instead, try to match sum with a non-neighbor edge
                # Or try product collision: val * v1 == val * v2 -> v1 == v2
                # Better: try to make val*v1 == some existing product
                for p in current_prods:
                    if v1 != 0 and p % v1 == 0:
                        cand = p // v1
                        if 1 <= cand <= 10000 and cand != old_val:
                            if cand not in used_values or cand == old_val:
                                candidates.append(cand)
            
            # Strategy 3: Random small perturbation
            if random.random() < 0.3 or not candidates:
                delta = random.randint(-50, 50)
                cand = old_val + delta
                if 1 <= cand <= 10000:
                    candidates.append(cand)
            
            if not candidates:
                continue
            
            # Deduplicate and limit
            candidates = list(set(candidates))
            if len(candidates) > 20:
                candidates = random.sample(candidates, 20)
            
            # Try each candidate and keep the best
            best_cand_score = current_score
            best_cand_val = old_val
            
            for cand in candidates:
                if cand in used_values and cand != old_val:
                    continue
                
                new_values = list(current_values)
                new_values[idx] = cand
                
                # Check distinctness
                if len(set(new_values)) != n:
                    continue
                
                new_score, new_sums, new_prods = fast_get_score(new_values, edge_list)
                
                if new_score < best_cand_score:
                    best_cand_score = new_score
                    best_cand_val = cand
                    best_cand_sums = new_sums
                    best_cand_prods = new_prods
            
            if best_cand_val != old_val:
                # Update state
                used_values.discard(old_val)
                used_values.add(best_cand_val)
                current_values[idx] = best_cand_val
                current_score = best_cand_score
                current_sums = best_cand_sums
                current_prods = best_cand_prods
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)
        
        return best_score, best_values

    def edge_local_search(values, edges, max_iters=20000):
        """Local search on edges only."""
        current_edges = set(edges)
        current_edge_list = list(current_edges)
        current_score, _, _ = fast_get_score(values, current_edge_list)
        
        best_score = current_score
        best_edges = set(current_edges)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try swapping one edge
            e_out_idx = random.randint(0, m - 1)
            e_out = current_edge_list[e_out_idx]
            
            # Find a non-edge to add
            while True:
                e_in = random.choice(all_edges)
                if e_in not in current_edges:
                    break
            
            new_edge_list = list(current_edge_list)
            new_edge_list[e_out_idx] = e_in
            
            new_score, _, _ = fast_get_score(values, new_edge_list)
            
            if new_score <= current_score:
                current_edges = set(new_edge_list)
                current_edge_list = new_edge_list
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
        
        return best_score, best_edges

    def value_local_search(values, edges, max_iters=15000):
        """Local search on values only."""
        current_values = list(values)
        edge_list = list(edges)
        current_score, _, _ = fast_get_score(current_values, edge_list)
        
        best_score = current_score
        best_values = list(current_values)
        
        used = set(current_values)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n - 1)
            
            if random.random() < 0.5:
                # Swap
                idx2 = random.randint(0, n - 1)
                if idx == idx2:
                    continue
                new_values = list(current_values)
                new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]
            else:
                # Perturb
                delta = random.randint(-30, 30)
                new_val = current_values[idx] + delta
                if not (1 <= new_val <= 10000):
                    continue
                if new_val in used and new_val != current_values[idx]:
                    continue
                new_values = list(current_values)
                new_values[idx] = new_val
            
            if len(set(new_values)) != n:
                continue
            
            new_score, _, _ = fast_get_score(new_values, edge_list)
            
            if new_score <= current_score:
                current_values = new_values
                current_score = new_score
                used = set(current_values)
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)
        
        return best_score, best_values

    def full_optimize(values, edges, time_budget=10.0):
        """Full optimization: alternate between edge and value search."""
        current_values = list(values)
        current_edges = set(edges)
        current_score, _, _ = fast_get_score(current_values, list(current_edges))
        
        best_score = current_score
        best_values = list(current_values)
        best_edges = set(current_edges)
        
        start = time.time()
        
        while time.time() - start < time_budget:
            # Edge optimization
            if time.time() > deadline:
                break
            e_budget = min(2.0, time_budget - (time.time() - start))
            if e_budget > 0.5:
                es, ee = edge_local_search(current_values, current_edges, max_iters=int(e_budget * 5000))
                if es < current_score:
                    current_score = es
                    current_edges = ee
                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)
            
            # Value optimization
            if time.time() > deadline:
                break
            v_budget = min(2.0, time_budget - (time.time() - start))
            if v_budget > 0.5:
                vs, vv = value_local_search(current_values, current_edges, max_iters=int(v_budget * 3000))
                if vs < current_score:
                    current_score = vs
                    current_values = vv
                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)
        
        return best_score, best_values, best_edges

    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None

    random.seed(42)
    
    # Generate diverse graph topologies
    graph_configs = []
    
    # 1. Near-regular graphs with different seeds
    for seed in range(5):
        g = generate_regular_graph(5, seed)
        graph_configs.append(g)
    
    # 2. Core-clique + spokes
    for seed in range(3):
        g = generate_complete_subgraph_plus_random(seed)
        graph_configs.append(g)
    
    # 3. Star + random
    for seed in range(3):
        g = generate_star_plus_clique(seed)
        graph_configs.append(g)
    
    # 4. Pure random
    for seed in range(3):
        rng = random.Random(seed)
        edges = set(rng.sample(all_edges, m))
        graph_configs.append(edges)
    
    # 5. Complete graph minus some edges (high density)
    for seed in range(2):
        rng = random.Random(seed + 100)
        all_e = list(all_edges)
        rng.shuffle(all_e)
        # Take first m edges (dense due to shuffle)
        edges = set(all_e[:m])
        graph_configs.append(edges)
    
    # Initial value configurations
    value_configs = [
        list(range(1, n + 1)),           # 1-14
        list(range(1, n + 1, 2)) + list(range(15, 27, 2)),  # odds + higher odds
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    ]
    
    # Random value configs
    for seed in range(4):
        rng = random.Random(seed + 200)
        vals = rng.sample(range(1, 1001), n)
        value_configs.append(vals)
    
    # Near-collision values (small range to encourage collisions)
    value_configs.append(list(range(1, 8)) + [10, 12, 15, 20, 25, 30, 40])
    value_configs.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    
    # Time allocation
    total_configs = len(graph_configs) * len(value_configs)
    time_per_config = 80.0 / total_configs
    time_per_config = max(time_per_config, 1.0)
    time_per_config = min(time_per_config, 5.0)
    
    for gi, edges in enumerate(graph_configs):
        if time.time() > deadline - 5.0:
            break
        
        for vi, vals in enumerate(value_configs):
            if time.time() > deadline - 2.0:
                break
            
            # Run collision-seeking optimization
            t0 = time.time()
            budget = min(time_per_config, deadline - time.time() - 2.0)
            if budget < 0.5:
                continue
            
            # First, do collision seeking
            max_iters = int(budget * 5000)
            max_iters = max(max_iters, 1000)
            max_iters = min(max_iters, 50000)
            
            cs_score, cs_values = collision_seeking_optimize(vals, edges, max_iters=max_iters)
            
            # Then full optimization
            if time.time() < deadline - 1.0:
                remaining = deadline - time.time() - 1.0
                remaining = min(remaining, budget * 0.5)
                if remaining > 0.5:
                    fs, fv, fe = full_optimize(cs_values, edges, time_budget=remaining)
                    if fs < cs_score:
                        cs_score = fs
                        cs_values = fv
                        edges = fe
            
            if cs_score < best_global_score:
                best_global_score = cs_score
                best_global_values = list(cs_values)
                best_global_edges = set(edges)
    
    # Final polish on best result
    if best_global_values is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        if remaining > 1.0:
            fs, fv, fe = full_optimize(best_global_values, best_global_edges, time_budget=remaining)
            if fs < best_global_score:
                best_global_score = fs
                best_global_values = fv
                best_global_edges = fe
    
    # Fallback
    if best_global_values is None:
        best_global_values = list(range(1, n + 1))
        best_global_edges = set(all_edges[:m])
        best_global_score = get_score(best_global_values, best_global_edges)
    
    # Ensure types
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
