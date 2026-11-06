import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_c5_exhaustive(n, adj):
        """Check if graph has a C5 subgraph (not induced)."""
        for combo in itertools.combinations(range(n), 5):
            degs = [0] * 5
            edge_count = 0
            for i in range(5):
                vi = combo[i]
                for j in range(i + 1, 5):
                    vj = combo[j]
                    if vj in adj[vi]:
                        degs[i] += 1
                        degs[j] += 1
                        edge_count += 1
            if edge_count == 5 and all(d == 2 for d in degs):
                return True
        return False

    def has_induced_c5(n, adj):
        """Check if graph has an induced C5."""
        for combo in itertools.combinations(range(n), 5):
            # Count edges within the 5 vertices
            edge_count = 0
            for i in range(5):
                vi = combo[i]
                for j in range(i + 1, 5):
                    vj = combo[j]
                    if vj in adj[vi]:
                        edge_count += 1
            # Induced C5 has exactly 5 edges
            if edge_count == 5:
                # Check each vertex has degree 2 within the induced subgraph
                degs = [0] * 5
                for i in range(5):
                    vi = combo[i]
                    for j in range(5):
                        if i != j and combo[j] in adj[vi]:
                            degs[i] += 1
                if all(d == 2 for d in degs):
                    return True
        return False

    def has_k9(n, adj):
        """Check if graph has a K9 clique."""
        # Use Bron-Kerbosch style or simple check
        # For n=33-35, check all 9-subsets is too slow
        # Use a smarter approach: find max clique
        # Simple approach: check if any 9-set is a clique
        # This is expensive but let's try with pruning
        
        def is_clique(vertices, adj):
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if vertices[j] not in adj[vertices[i]]:
                        return False
            return True
        
        # Use a heuristic: try to build K9 greedily
        # Or use the fact that we can check via complement
        # For now, use a randomized approach with some structure
        
        # Try to find K9 by building it incrementally
        def try_build_k9(adj, n, attempt=0):
            if attempt > 50:
                return False
            # Pick a random starting vertex
            start = random.randint(0, n-1)
            candidates = set(range(n)) - {start}
            # Filter to neighbors of start
            candidates = candidates & adj[start]
            if len(candidates) < 8:
                return try_build_k9(adj, n, attempt+1)
            
            # Build clique incrementally
            clique = [start]
            while len(clique) < 9 and candidates:
                # Pick a random candidate that connects to all in clique
                valid = []
                for v in candidates:
                    if all(u in adj[v] for u in clique):
                        valid.append(v)
                if not valid:
                    break
                v = random.choice(valid)
                clique.append(v)
                candidates = candidates & adj[v]
            
            if len(clique) >= 9:
                return True
            return try_build_k9(adj, n, attempt+1)
        
        return try_build_k9(adj, n)

    def find_large_independent_set(n, adj):
        remaining = set(range(n))
        ind_set = []
        while remaining:
            min_deg = float('inf')
            v = None
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
            ind_set.append(v)
            remaining.remove(v)
            remaining -= adj[v]
        return ind_set

    def complement_edges(n, edges):
        """Get complement of a graph."""
        comp = []
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in edge_set:
                    comp.append([u, v])
        return comp

    def validate_original(n, edges):
        """Validate the original graph G: no C5, no independent set of 9."""
        adj = build_adj(n, edges)
        if has_c5_exhaustive(n, adj):
            return False
        ind_set = find_large_independent_set(n, adj)
        if len(ind_set) >= 9:
            return False
        return True

    # Baseline: n=32, 8 disjoint K4s
    best_result = None
    n_base = 32
    edges_base = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])
    best_result = {'n': n_base, 'edges': edges_base}
    
    # Try to find valid graphs for n=33, 34, 35 using complement GA
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 30:
            break
            
        # Genetic algorithm on complement H
        # H has no induced C5 and no K9
        # G = complement(H) must have no C5 and alpha < 9
        
        pop_size = 20
        generations = 30
        
        # Initialize population with random graphs
        population = []
        for _ in range(pop_size):
            h_edges = set()
            for u in range(target_n):
                for v in range(u+1, target_n):
                    if random.random() < 0.5:
                        h_edges.add((u, v))
            population.append(list(h_edges))
        
        found_valid = False
        
        for gen in range(generations):
            if time.time() > deadline - 5:
                break
            
            # Evaluate fitness
            def fitness(h_edges_list):
                # Convert to adjacency
                h_adj = build_adj(target_n, h_edges_list)
                
                # Check induced C5 in H (this means C5 in G)
                # Actually, we need: G has no C5 subgraph
                # G = complement(H)
                # So we need to check C5 in G, not H
                g_edges = complement_edges(target_n, h_edges_list)
                g_adj = build_adj(target_n, g_edges)
                
                # Check C5 in G
                c5_exists = has_c5_exhaustive(target_n, g_adj)
                
                # Check alpha in G < 9
                ind_set = find_large_independent_set(target_n, g_adj)
                alpha_violation = len(ind_set) >= 9
                
                # Fitness: lower is better
                score = 0
                if c5_exists:
                    score += 100
                if alpha_violation:
                    score += 100
                # Prefer denser H (sparser G) - this helps avoid C5
                # Actually, we want G to be valid, so let's just use constraint violations
                return score
            
            # Evaluate all
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(key=lambda x: x[0])
            
            # Check if any is valid
            for score, ind in scored:
                if score == 0:
                    # Verify thoroughly
                    g_edges = complement_edges(target_n, ind)
                    if validate_original(target_n, g_edges):
                        best_result = {'n': target_n, 'edges': g_edges}
                        found_valid = True
                        break
            if found_valid:
                break
            
            # Selection: keep top half
            top_half = [ind for _, ind in scored[:pop_size//2]]
            
            # Crossover and mutation
            new_pop = list(top_half)
            while len(new_pop) < pop_size:
                if len(top_half) < 2:
                    # Just add random
                    h_edges = set()
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            if random.random() < 0.5:
                                h_edges.add((u, v))
                    new_pop.append(list(h_edges))
                else:
                    p1 = random.choice(top_half)
                    p2 = random.choice(top_half)
                    # Uniform crossover
                    child = []
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            if random.random() < 0.5:
                                if (u, v) in p1:
                                    child.append((u, v))
                            else:
                                if (u, v) in p2:
                                    child.append((u, v))
                    # Mutation
                    for i in range(len(child)):
                        if random.random() < 0.1:
                            u, v = child[i]
                            if random.random() < 0.5:
                                child.pop(i)
                            else:
                                # Add random edge
                                eu, ev = random.sample(range(target_n), 2)
                                eu, ev = min(eu, ev), max(eu, ev)
                                if (eu, ev) not in child:
                                    child.append((eu, ev))
                            break
                    new_pop.append(child)
            
            population = new_pop
        
        if found_valid:
            break
    
    return best_result
