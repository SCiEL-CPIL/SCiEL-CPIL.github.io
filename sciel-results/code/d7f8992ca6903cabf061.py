import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            P_X = P.union(X)
            if not P_X:
                return
            
            max_deg = -1
            pivot = None
            for u in P_X:
                deg = len(P.intersection(adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P - adj[pivot]):
                bk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
            
        m = len(cliques)
        
        # Precompute mappings
        vertex_to_cliques = [[] for _ in range(n)]
        clique_to_vertices = [set(c) for c in cliques]
        
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Precompute bitmask of cliques covered by each vertex
        # For fast intersection checks
        vertex_clique_mask = [0] * n
        for v in range(n):
            mask = 0
            for ci in vertex_to_cliques[v]:
                mask |= (1 << ci)
            vertex_clique_mask[v] = mask
        
        best_solution = None
        best_size = float('inf')
        
        # Number of restarts - keep reasonable given time budget
        restarts = 300
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            uncovered = set(range(m))
            solution = set()
            
            # Greedy Phase with randomization among ties
            while uncovered:
                best_count = -1
                candidates = []
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        candidates = [v]
                    elif count == best_count and count > 0:
                        candidates.append(v)
                        
                if not candidates:
                    break
                    
                v = random.choice(candidates)
                solution.add(v)
                
                for ci in vertex_to_cliques[v]:
                    uncovered.discard(ci)
                    
            # Local Search Phase
            improved = True
            local_iter = 0
            while improved and local_iter < 50:
                improved = False
                local_iter += 1
                
                # 1. Redundancy Removal using coverage counts
                coverage = [0] * m
                for v in solution:
                    for ci in vertex_to_cliques[v]:
                        coverage[ci] += 1
                
                verts_to_check = list(solution)
                random.shuffle(verts_to_check)
                for v in verts_to_check:
                    if v not in solution:
                        continue
                    redundant = True
                    for ci in vertex_to_cliques[v]:
                        if coverage[ci] <= 1:
                            redundant = False
                            break
                    if redundant:
                        solution.remove(v)
                        for ci in vertex_to_cliques[v]:
                            coverage[ci] -= 1
                        improved = True
                        break
                
                if improved:
                    continue
                
                # 2. 2-for-1 Exchange: remove two vertices, add one
                sol_list = list(solution)
                random.shuffle(sol_list)
                found_improvement = False
                for i in range(len(sol_list)):
                    if time.time() > deadline - 5.0:
                        break
                    v1 = sol_list[i]
                    if v1 not in solution:
                        continue
                    for j in range(i + 1, len(sol_list)):
                        v2 = sol_list[j]
                        if v2 not in solution:
                            continue
                        
                        # Find cliques that become uncovered if we remove v1 and v2
                        critical_cliques = set()
                        
                        # Check v1's cliques
                        for ci in vertex_to_cliques[v1]:
                            if coverage[ci] == 1:
                                critical_cliques.add(ci)
                            elif coverage[ci] == 2 and v2 in clique_to_vertices[ci]:
                                critical_cliques.add(ci)
                                
                        # Check v2's cliques for cases where v1 not in ci
                        for ci in vertex_to_cliques[v2]:
                            if coverage[ci] == 1 and v1 not in clique_to_vertices[ci]:
                                critical_cliques.add(ci)
                        
                        if not critical_cliques:
                            # Removing v1 and v2 uncovers nothing
                            solution.remove(v1)
                            solution.remove(v2)
                            improved = True
                            found_improvement = True
                            break
                        
                        # Find a vertex u not in solution that covers ALL critical cliques
                        # Use bitmask intersection for efficiency
                        target_mask = 0
                        for ci in critical_cliques:
                            target_mask |= (1 << ci)
                        
                        best_u = None
                        for u in range(n):
                            if u in solution:
                                continue
                            # Check if u covers all critical cliques using bitmask
                            if (vertex_clique_mask[u] & target_mask) == target_mask:
                                best_u = u
                                break
                                
                        if best_u is not None:
                            solution.remove(v1)
                            solution.remove(v2)
                            solution.add(best_u)
                            improved = True
                            found_improvement = True
                            break
                        
                    if found_improvement:
                        break
                        
                # 3. 1-for-1 Exchange: remove one vertex, add another
                if not improved:
                    sol_list = list(solution)
                    random.shuffle(sol_list)
                    for v in sol_list:
                        if v not in solution:
                            continue
                        
                        # Find cliques that become uncovered if we remove v
                        critical_cliques = set()
                        for ci in vertex_to_cliques[v]:
                            if coverage[ci] == 1:
                                critical_cliques.add(ci)
                        
                        if not critical_cliques:
                            # v is redundant, should have been caught above
                            solution.remove(v)
                            improved = True
                            break
                        
                        # Find a vertex u not in solution that covers ALL critical cliques
                        target_mask = 0
                        for ci in critical_cliques:
                            target_mask |= (1 << ci)
                        
                        best_u = None
                        for u in range(n):
                            if u in solution:
                                continue
                            if (vertex_clique_mask[u] & target_mask) == target_mask:
                                best_u = u
                                break
                                
                        if best_u is not None:
                            solution.remove(v)
                            solution.add(best_u)
                            improved = True
                            break

            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
                
        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10.0:
            # Fallback: return all touched vertices
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u, v)}
            out.append(sorted(touched))
        else:
            out.append(solve_instance(n, p, seed))
            
    return out
