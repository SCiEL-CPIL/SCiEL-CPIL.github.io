import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, adj, n):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        
        if P:
            pivot = max(P, key=lambda p: len(P & adj[p]))
        elif X:
            pivot = max(X, key=lambda p: len(P & adj[p]))
        else:
            return
            
        for v in list(P - adj[pivot]):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X, adj, n)
            P.remove(v)
            X.add(v)

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        C = len(valid_cliques)
        
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        best_set = None
        best_size = float('inf')
        
        rng_solver = random.Random(seed + 123)
        
        # Strategy 1: Vertex Cover on Clique Intersection Graph
        # Build intersection graph: nodes are cliques, edge if they share a vertex
        clique_adj = [set() for _ in range(C)]
        for i in range(C):
            for j in range(i+1, C):
                if valid_cliques[i] & valid_cliques[j]:
                    clique_adj[i].add(j)
                    clique_adj[j].add(i)
        
        # Find maximum matching and derive vertex cover
        # We use a simple greedy + augmenting path approach for the general graph
        match = [-1] * C
        visited = [False] * C
        
        def find_augmenting_path(u, seen):
            for v in clique_adj[u]:
                if not seen[v]:
                    seen[v] = True
                    if match[v] == -1 or find_augmenting_path(match[v], seen):
                        match[u] = v
                        match[v] = u
                        return True
            return False
        
        for u in range(C):
            if match[u] == -1:
                seen = [False] * C
                find_augmenting_path(u, seen)
                
        vc_cliques = set()
        for u in range(C):
            if match[u] != -1:
                vc_cliques.add(u)
                vc_cliques.add(match[u])
        
        # For each clique in vertex cover, we need to select vertices
        # Greedily select vertices to cover all cliques in vc_cliques
        hit_cliques = set()
        current_set = set()
        
        # Prioritize cliques in the vertex cover
        cliques_to_cover = list(vc_cliques)
        # Add any remaining cliques not in vertex cover
        for i in range(C):
            if i not in vc_cliques:
                cliques_to_cover.append(i)
        
        for ci in cliques_to_cover:
            if ci in hit_cliques:
                continue
            # Find vertex in this clique that covers most uncovered cliques
            best_v = -1
            max_hits = -1
            candidates = []
            for v in valid_cliques[ci]:
                hits = 0
                for cj in vertex_cliques[v]:
                    if cj not in hit_cliques:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_v = v
                    candidates = [v]
                elif hits == max_hits and hits > 0:
                    candidates.append(v)
            
            if candidates:
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for cj in vertex_cliques[v]:
                    hit_cliques.add(cj)
        
        # Local search: reduction
        improved = True
        while improved:
            improved = False
            vs = list(current_set)
            rng_solver.shuffle(vs)
            for v in vs:
                if time.time() > deadline:
                    break
                can_remove = True
                for ci in vertex_cliques[v]:
                    c = valid_cliques[ci]
                    found_other = False
                    for u in c:
                        if u != v and u in current_set:
                            found_other = True
                            break
                    if not found_other:
                        can_remove = False
                        break
                if can_remove:
                    current_set.remove(v)
                    improved = True
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # Strategy 2: Standard Greedy with random restarts
        num_restarts = 500
        
        for run in range(num_restarts):
            if time.time() > deadline:
                break
                
            hit_cliques = set()
            current_set = set()
            
            while len(hit_cliques) < C:
                best_v = -1
                max_hits = -1
                candidates = []
                
                for v in range(n):
                    if v in current_set:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)
                        
                if max_hits == 0:
                    break
                    
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
        # Strategy 3: Exact branch-and-bound with time limit
        # This can find the true optimum for small instances
        if time.time() < deadline - 10 and C <= 200:
            # Precompute for speed: for each vertex, which cliques it covers
            v_cliques_list = [list(vertex_cliques[v]) if v in vertex_cliques else [] for v in range(n)]
            
            # Lower bound: for each uncovered clique, we need at least 1 more vertex
            # Better LB: greedy packing of disjoint cliques
            def compute_lb(uncovered):
                if not uncovered:
                    return 0
                # Greedy: find disjoint cliques
                remaining = set(uncovered)
                lb = 0
                while remaining:
                    # Pick the smallest clique (most constrained)
                    ci = min(remaining, key=lambda i: len(valid_cliques[i]))
                    lb += 1
                    # Remove all cliques that intersect this one
                    to_remove = set()
                    for cj in remaining:
                        if valid_cliques[ci] & valid_cliques[cj]:
                            to_remove.add(cj)
                    remaining -= to_remove
                return lb
            
            # Branch and bound
            uncovered = set(range(C))
            current_solution = set()
            best_bb_size = best_size
            best_bb_set = None
            
            # Order vertices by how many cliques they cover (descending) for better pruning
            vertex_order = sorted(range(n), key=lambda v: len(v_cliques_list[v]), reverse=True)
            
            def bb(idx, uncovered_frozenset, current_size):
                nonlocal best_bb_size, best_bb_set
                if time.time() > deadline:
                    return
                if current_size >= best_bb_size:
                    return
                if not uncovered_frozenset:
                    if current_size < best_bb_size:
                        best_bb_size = current_size
                        best_bb_set = set(current_solution)
                    return
                
                # Lower bound
                lb = compute_lb(uncovered_frozenset)
                if current_size + lb >= best_bb_size:
                    return
                
                # Pick a clique to branch on
                # Choose the most constrained uncovered clique
                ci = min(uncovered_frozenset, key=lambda i: len(valid_cliques[i]))
                clique = valid_cliques[ci]
                
                # Try each vertex in this clique
                # Order by number of additional cliques covered
                candidates = sorted(clique, key=lambda v: len([cj for cj in v_cliques_list[v] if cj in uncovered_frozenset]), reverse=True)
                
                for v in candidates:
                    if time.time() > deadline:
                        return
                    new_uncovered = uncovered_frozenset - frozenset(v_cliques_list[v])
                    current_solution.add(v)
                    bb(idx + 1, new_uncovered, current_size + 1)
                    current_solution.remove(v)
                    if current_size + 1 >= best_bb_size:
                        break
                # Also branch: don't pick any vertex from this clique? No, we MUST hit it.
                # So we must pick at least one. The above tries each.
                # But we also need to consider that we might pick a vertex not in this clique
                # that hits it... no, the vertex must be IN the clique to hit it.
                # So branching on vertices of the clique is complete.
            
            bb(0, frozenset(range(C)), 0)
            
            if best_bb_set is not None and best_bb_size < best_size:
                best_size = best_bb_size
                best_set = best_bb_set
        
        return sorted(best_set) if best_set else []

    results = []
    # Allocate time budget: give more time to harder instances
    instance_budgets = [45, 45, 45]  # Total 135s, with 5s safety margin
    
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        remaining = deadline - time.time()
        if remaining < 5:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        # Temporarily adjust deadline for this instance
        global_deadline = deadline
        deadline = min(deadline, time.time() + min(instance_budgets[idx], remaining - 2))
        
        res = solve_instance(n, p, seed)
        results.append(res)
        
        deadline = global_deadline
        
    return results
