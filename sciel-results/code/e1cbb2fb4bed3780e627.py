import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch_maximal_cliques(adj):
        """
        Enumerates all maximal cliques in the graph defined by adjacency lists.
        Uses Bron-Kerbosch with pivot.
        """
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X
            # Heuristic: choose u with max degree in P
            P_X = P.union(X)
            if not P_X:
                return
            u = max(P_X, key=lambda v: len(adj[v].intersection(P)))
            
            # Iterate over P \ N(u)
            for v in list(P - adj[u]):
                R_new = R.union({v})
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bk(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
        
        # Initial call
        # P = all vertices, X = empty
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj, n, deadline):
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        # Convert to list of sets for easier manipulation
        cliques_list = [set(c) for c in cliques]
        num_cliques = len(cliques_list)
        
        # Precompute vertex -> list of clique indices
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Use bitmask for coverage: bit i set means clique i is covered
        # For n <= 40 and typically < 1000 cliques, this is feasible
        
        def greedy_init():
            S = set()
            cover_mask = 0  # Bitmask of covered cliques
            remaining_cliques = num_cliques
            
            while remaining_cliques > 0:
                best_v = -1
                max_new = -1
                for v in range(n):
                    if v in S:
                        continue
                    new_cover = 0
                    for ci in vertex_to_cliques[v]:
                        if not (cover_mask & (1 << ci)):
                            new_cover += 1
                    if new_cover > max_new:
                        max_new = new_cover
                        best_v = v
                if best_v == -1:
                    break
                S.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    if not (cover_mask & (1 << ci)):
                        remaining_cliques -= 1
                        cover_mask |= (1 << ci)
            return S

        def prune(S):
            S = set(S)
            cover_count = [0] * num_cliques
            for v in S:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    # Check if removing v keeps all cliques covered
                    can_remove = True
                    for ci in vertex_to_cliques[v]:
                        if cover_count[ci] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        S.remove(v)
                        for ci in vertex_to_cliques[v]:
                            cover_count[ci] -= 1
                        changed = True
            return S

        def local_search(S, rng, deadline):
            """Local search with removals and swaps"""
            current_S = set(S)
            current_size = len(current_S)
            
            # Precompute cover counts
            cover_count = [0] * num_cliques
            for v in current_S:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            
            no_improve = 0
            max_no_improve = 5000
            
            while time.time() < deadline:
                if no_improve > max_no_improve:
                    break
                
                # Try removals first
                removed = False
                if current_S:
                    # Try to remove a random vertex
                    v_out = rng.choice(list(current_S))
                    is_critical = False
                    for ci in vertex_to_cliques[v_out]:
                        if cover_count[ci] == 1:
                            is_critical = True
                            break
                    
                    if not is_critical:
                        current_S.remove(v_out)
                        for ci in vertex_to_cliques[v_out]:
                            cover_count[ci] -= 1
                        current_size -= 1
                        no_improve = 0
                        removed = True
                
                if not removed:
                    # Try swaps: replace v_in with v_out
                    if len(current_S) > 0:
                        v_in = rng.choice(list(current_S))
                        
                        # Find cliques uniquely covered by v_in
                        unique_cliques = [ci for ci in vertex_to_cliques[v_in] 
                                         if cover_count[ci] == 1]
                        
                        if not unique_cliques:
                            # Can remove v_in directly
                            current_S.remove(v_in)
                            for ci in vertex_to_cliques[v_in]:
                                cover_count[ci] -= 1
                            current_size -= 1
                            no_improve = 0
                            continue
                        
                        # Find v_out that covers all unique cliques
                        valid_v_outs = []
                        for v_out in range(n):
                            if v_out in current_S:
                                continue
                            ok = True
                            v_out_cliques_set = set(vertex_to_cliques[v_out])
                            for ci in unique_cliques:
                                if ci not in v_out_cliques_set:
                                    ok = False
                                    break
                            if ok:
                                valid_v_outs.append(v_out)
                        
                        if valid_v_outs:
                            v_out = rng.choice(valid_v_outs)
                            # Perform swap
                            current_S.remove(v_in)
                            for ci in vertex_to_cliques[v_in]:
                                cover_count[ci] -= 1
                            current_S.add(v_out)
                            for ci in vertex_to_cliques[v_out]:
                                cover_count[ci] += 1
                            # Size unchanged, but may enable future removals
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
            
            return current_S

        # Multiple restarts with different initializations
        best_S = greedy_init()
        best_S = prune(best_S)
        best_size = len(best_S)
        
        # Local search from greedy solution
        rng = random.Random(42)
        current_time = time.time()
        time_per_restart = (deadline - current_time) / 10  # 10 restarts
        restart_deadline = current_time + time_per_restart
        
        if time.time() < restart_deadline:
            improved_S = local_search(best_S, rng, restart_deadline)
            if len(improved_S) < best_size:
                best_S = improved_S
                best_size = len(best_S)
        
        # More aggressive local search with longer time
        if time.time() < deadline:
            rng2 = random.Random(123)
            improved_S = local_search(best_S, rng2, deadline)
            if len(improved_S) < best_size:
                best_S = improved_S
                best_size = len(best_S)
        
        return sorted(best_S)

    out = []
    # Allocate time roughly equally, but give more to larger instances?
    # 30, 35, 40 nodes.
    # Let's just run until deadline.
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > deadline:
            # Fallback: return empty or best effort? 
            # If we run out of time, we should return what we have or a valid simple set.
            # The greedy init is valid. Let's just run greedy if SA didn't start or if we are late.
            adj, edges = random_graph(n, p, seed)
            # Quick greedy
            cliques = bron_kerbosch_maximal_cliques(adj)
            S = set()
            remaining = [set(c) for c in cliques]
            while remaining:
                best_v = None
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = sum(1 for c in remaining if v in c)
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v is None: break
                S.add(best_v)
                remaining = [c for c in remaining if best_v not in c]
            out.append(sorted(S))
            continue
            
        adj, edges = random_graph(n, p, seed)
        # Pass remaining time
        res = solve_instance(adj, n, deadline)
        out.append(res)
        
    return out
