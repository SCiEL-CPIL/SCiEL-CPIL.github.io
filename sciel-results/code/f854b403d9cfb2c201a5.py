import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin for 150s kill

    def remaining_time():
        return DEADLINE - (time.time() - t0)

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_max_cliques(adj, n):
        max_cliques = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            union = P | X
            if not union:
                return
            best_u = max(union, key=lambda u: len(P & adj[u]))
            candidates = P - adj[best_u]
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return max_cliques

    def greedy_hit_set(n, cliques):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        # vertex -> frozenset of clique indices
        v2c = {}
        for i, cl in enumerate(cliques):
            for v in cl:
                if v not in v2c:
                    v2c[v] = set()
                v2c[v].add(i)
        uncovered = set(range(len(cliques)))
        chosen = []
        while uncovered:
            best_v = None
            best_cnt = -1
            for v in v2c:
                if v in chosen:
                    continue
                cnt = len(v2c[v] & uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None:
                break
            chosen.append(best_v)
            uncovered -= v2c[best_v]
        return chosen

    def local_search_2swap(n, cliques, sol):
        """Try 2-swap: replace two selected vertices with two others that cover more."""
        sol_set = set(sol)
        # Precompute: for each vertex, which cliques it covers
        v2c = {}
        for i, cl in enumerate(cliques):
            for v in cl:
                if v not in v2c:
                    v2c[v] = set()
                v2c[v].add(i)
        
        covered = set()
        for v in sol:
            covered |= v2c[v]
        
        # Uncovered cliques
        uncovered = set(range(len(cliques))) - covered
        if not uncovered:
            return sol
        
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None
            for i, v1 in enumerate(sol):
                for v2 in sol[i+1:]:
                    # Try removing v1, v2 and adding two new vertices
                    # This is expensive; do a simpler 1-swap first
                    pass
            
            # 1-swap: remove one, add one that covers more uncovered
            for v_remove in sol:
                for v_add in range(n):
                    if v_add in sol_set:
                        continue
                    gain = len(v2c.get(v_add, set()) & uncovered) - 1
                    # Actually we need to check if removing v_remove leaves any clique uncovered
                    # that v_add doesn't cover
                    # Simpler: just try if v_add covers something v_remove doesn't
                    if v_add in v2c and v_remove in v2c:
                        # Check if v_remove covers any clique that no other sol vertex covers
                        v_remove_only = v2c[v_remove] - covered
                        if v_remove_only:
                            # v_add must cover those
                            if not (v_remove_only <= (v2c[v_add] - covered)):
                                continue
                        new_gain = len(v2c[v_add] & uncovered)
                        if new_gain > 0:
                            # Try the swap
                            new_sol = [v for v in sol if v != v_remove] + [v_add]
                            new_covered = set()
                            for v in new_sol:
                                new_covered |= v2c[v]
                            if len(new_covered) > len(covered):
                                sol = new_sol
                                sol_set = set(sol)
                                covered = new_covered
                                uncovered = set(range(len(cliques))) - covered
                                improved = True
                                break
                if improved:
                    break
        return sol

    def solve_exact_bnb(n, cliques, time_budget):
        """Exact branch-and-bound for hitting set."""
        if not cliques:
            return []
        
        # Build vertex -> cliques mapping
        v2c = {}
        for i, cl in enumerate(cliques):
            for v in cl:
                if v not in v2c:
                    v2c[v] = set()
                v2c[v].add(i)
        
        # Dominance pruning: if S_u ⊆ S_v, u is dominated
        vertices_in_cliques = set()
        for cl in cliques:
            vertices_in_cliques |= cl
        
        dominated = set()
        for u in vertices_in_cliques:
            if u in dominated:
                continue
            Su = v2c[u]
            for v in vertices_in_cliques:
                if u == v or v in dominated:
                    continue
                if Su <= v2c[v]:
                    dominated.add(u)
                    break
        
        active = [v for v in vertices_in_cliques if v not in dominated]
        
        # Find forced vertices
        forced = set()
        remaining = set(range(len(cliques)))
        changed = True
        while changed:
            changed = False
            for i in list(remaining):
                cl = cliques[i]
                active_in_cl = [v for v in cl if v not in dominated and v not in forced]
                if len(active_in_cl) == 1:
                    v = active_in_cl[0]
                    forced.add(v)
                    for ci in v2c[v]:
                        remaining.discard(ci)
                    changed = True
        
        if not remaining:
            return sorted(forced)
        
        # B&B
        best_size = len(greedy_hit_set(n, cliques))
        best_sol = set()
        
        # Use bitmask for uncovered cliques if manageable
        num_cliques = len(cliques)
        if num_cliques <= 60:
            # Bitmask approach
            def bb(uncovered_mask, cur_size, cur_sol):
                nonlocal best_size, best_sol
                if time.time() - t0 > DEADLINE - 5:
                    return
                if uncovered_mask == 0:
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol = set(cur_sol)
                    return
                
                # Lower bound: at least 1
                if cur_size + 1 >= best_size:
                    return
                
                # Find clique with fewest available vertices
                min_cnt = float('inf')
                branch_clique = -1
                for i in range(num_cliques):
                    if uncovered_mask & (1 << i):
                        cnt = 0
                        for v in cliques[i]:
                            if v not in dominated and v not in cur_sol:
                                cnt += 1
                        if cnt < min_cnt:
                            min_cnt = cnt
                            branch_clique = i
                
                if branch_clique == -1:
                    return
                
                branch_cl = cliques[branch_clique]
                verts = [v for v in branch_cl if v not in dominated and v not in cur_sol]
                verts.sort(key=lambda v: sum(1 for i in range(num_cliques) if uncovered_mask & (1 << i) and i in v2c.get(v, set())), reverse=True)
                
                for v in verts:
                    new_mask = uncovered_mask
                    for ci in v2c[v]:
                        new_mask &= ~(1 << ci)
                    bb(new_mask, cur_size + 1, cur_sol | {v})
                    if time.time() - t0 > DEADLINE - 5:
                        return
            
            init_mask = 0
            for i in remaining:
                init_mask |= (1 << i)
            bb(init_mask, len(forced), set(forced))
        else:
            # Set-based B&B
            def bb(uncovered, cur_size, cur_sol):
                nonlocal best_size, best_sol
                if time.time() - t0 > DEADLINE - 5:
                    return
                if not uncovered:
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol = set(cur_sol)
                    return
                
                if cur_size + 1 >= best_size:
                    return
                
                min_cnt = float('inf')
                branch_clique = -1
                for i in uncovered:
                    cnt = 0
                    for v in cliques[i]:
                        if v not in dominated and v not in cur_sol:
                            cnt += 1
                    if cnt < min_cnt:
                        min_cnt = cnt
                        branch_clique = i
                
                if branch_clique == -1:
                    return
                
                branch_cl = cliques[branch_clique]
                verts = [v for v in branch_cl if v not in dominated and v not in cur_sol]
                verts.sort(key=lambda v: len(v2c.get(v, set()) & uncovered), reverse=True)
                
                for v in verts:
                    new_uncov = set(uncovered)
                    for ci in v2c[v]:
                        new_uncov.discard(ci)
                    bb(new_uncov, cur_size + 1, cur_sol | {v})
                    if time.time() - t0 > DEADLINE - 5:
                        return
            
            bb(remaining, len(forced), set(forced))
        
        if best_sol:
            return sorted(best_sol)
        return greedy_hit_set(n, cliques)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() - t0 > DEADLINE - 5:
            results.append([])
            continue
        
        edges = instance(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch_max_cliques(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        time_spent = time.time() - t0
        remaining = DEADLINE - time_spent
        budget = remaining / (len(instances) - idx)
        
        sol = solve_exact_bnb(n, cliques, budget)
        results.append(sol)
    
    return results
