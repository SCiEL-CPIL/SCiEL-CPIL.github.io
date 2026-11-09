import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145  # safety margin from 150s

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def bron_kerbosch_max_clique(adj, n, rng=None):
        """Find a maximum (or near-maximum) clique using Bron-Kerbosch with pivot.
        Returns the largest clique found."""
        best_clique = []
        
        def bk(R_mask, P_mask, X_mask):
            nonlocal best_clique
            if P_mask == 0 and X_mask == 0:
                # R is a maximal clique
                if R_mask.bit_count() > len(best_clique):
                    best_clique = list(iter_bits(R_mask))
                return
            
            # Choose pivot u from P ∪ X with maximum |P ∩ N(u)|
            PX = P_mask | X_mask
            best_pivot = -1
            best_count = -1
            for u in iter_bits(PX):
                count = (P_mask & adj[u]).bit_count()
                if count > best_count:
                    best_count = count
                    best_pivot = u
            
            if best_pivot == -1:
                return
            
            # Iterate over vertices in P \\ N(pivot)
            candidates = P_mask & ~(1 << best_pivot) & ~adj[best_pivot]
            if candidates == 0:
                candidates = P_mask
            
            for v in list(iter_bits(candidates)):
                v_bit = 1 << v
                new_R = R_mask | v_bit
                new_P = P_mask & adj[v]
                new_X = X_mask & adj[v]
                bk(new_R, new_P, new_X)
                P_mask &= ~v_bit
                X_mask |= v_bit
        
        bk(0, (1 << n) - 1, 0)
        return best_clique

    def bron_kerbosch_all_maximal(adj, n, max_cliques=1000):
        """Find all maximal cliques (up to max_cliques)."""
        cliques = []
        
        def bk(R_mask, P_mask, X_mask):
            nonlocal cliques
            if P_mask == 0 and X_mask == 0:
                cliques.append(list(iter_bits(R_mask)))
                return
            if len(cliques) >= max_cliques:
                return
            
            PX = P_mask | X_mask
            best_pivot = -1
            best_count = -1
            for u in iter_bits(PX):
                count = (P_mask & adj[u]).bit_count()
                if count > best_count:
                    best_count = count
                    best_pivot = u
            
            if best_pivot == -1:
                return
            
            candidates = P_mask & ~adj[best_pivot]
            if candidates == 0:
                candidates = P_mask
            
            for v in list(iter_bits(candidates)):
                v_bit = 1 << v
                new_R = R_mask | v_bit
                new_P = P_mask & adj[v]
                new_X = X_mask & adj[v]
                bk(new_R, new_P, new_X)
                P_mask &= ~v_bit
                X_mask |= v_bit
                if len(cliques) >= max_cliques:
                    return
        
        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(n, p, seed, time_budget):
        adj_orig = get_graph(n, p, seed)

        if not any(adj_orig):
            return []

        def validate(cliques, adj_mask_orig, n):
            """Validate that cliques form a proper edge partition."""
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
            return True

        def remove_clique_edges(C, adj):
            """Remove all edges within clique C from adj."""
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def greedy_max_clique_partition(adj_mask_orig, n, deadline):
            """Repeatedly find a large clique, remove its edges, repeat."""
            adj = list(adj_mask_orig)
            cliques = []
            
            while time.time() < deadline:
                # Find edges remaining
                has_edges = False
                for u in range(n):
                    if adj[u]:
                        has_edges = True
                        break
                if not has_edges:
                    break
                
                # Find a large clique using Bron-Kerbosch
                C = bron_kerbosch_max_clique(adj, n)
                if len(C) < 2:
                    # No clique of size >= 2, take any edge
                    for u in range(n):
                        if adj[u]:
                            v = list(iter_bits(adj[u]))[0]
                            C = [u, v]
                            break
                    if len(C) < 2:
                        break
                
                cliques.append(C)
                remove_clique_edges(C, adj)
            
            return cliques

        def greedy_with_all_maximal(adj_mask_orig, n, deadline):
            """Use all maximal cliques found, pick best ones greedily."""
            adj = list(adj_mask_orig)
            cliques = []
            
            while time.time() < deadline:
                has_edges = False
                for u in range(n):
                    if adj[u]:
                        has_edges = True
                        break
                if not has_edges:
                    break
                
                # Find all maximal cliques in current graph
                max_cliques = bron_kerbosch_all_maximal(adj, n, max_cliques=500)
                
                if not max_cliques:
                    for u in range(n):
                        if adj[u]:
                            v = list(iter_bits(adj[u]))[0]
                            cliques.append([u, v])
                            remove_clique_edges([u, v], adj)
                            break
                    continue
                
                # Pick the largest clique
                max_cliques.sort(key=lambda c: -len(c))
                C = max_cliques[0]
                if len(C) < 2:
                    C = [list(iter_bits(adj[u]))[0], u] if adj[u] else []
                    if len(C) < 2:
                        break
                
                cliques.append(C)
                remove_clique_edges(C, adj)
            
            return cliques

        def merge_compatible(cliques, adj_mask_orig, n, deadline):
            """Try to merge pairs of cliques that together form a clique."""
            def is_clique_union(C1, C2, adj):
                for u in C1:
                    for v in C2:
                        if not (adj[u] & (1 << v)):
                            return False
                return True
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_cliques = []
                i = 0
                while i < len(cliques):
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        if is_clique_union(cliques[i], cliques[j], adj_mask_orig):
                            new_cliques.append(cliques[i] + cliques[j])
                            changed = True
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(cliques[i])
                    i += 1
                cliques = new_cliques
            return cliques

        best_cliques = None
        best_count = float('inf')

        deadline = start_time + time_budget - 2.0

        # Strategy 1: Greedy max clique
        t1 = min(deadline, time.time() + time_budget * 0.35)
        c1 = greedy_max_clique_partition(adj_orig, n, t1)
        if validate(c1, adj_orig, n):
            c1 = merge_compatible(c1, adj_orig, n, deadline)
            if validate(c1, adj_orig, n) and len(c1) < best_count:
                best_count = len(c1)
                best_cliques = c1

        # Strategy 2: Greedy with all maximal cliques
        t2 = min(deadline, time.time() + time_budget * 0.35)
        c2 = greedy_with_all_maximal(adj_orig, n, t2)
        if validate(c2, adj_orig, n):
            c2 = merge_compatible(c2, adj_orig, n, deadline)
            if validate(c2, adj_orig, n) and len(c2) < best_count:
                best_count = len(c2)
                best_cliques = c2

        # Strategy 3: Randomized restarts with perturbation
        remaining = deadline - time.time()
        if remaining > 2:
            rng_master = random.Random(seed + 7777)
            num_restarts = max(2, int(remaining / 1.0))
            for r in range(num_restarts):
                if time.time() > deadline:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                adj = list(adj_orig)
                cliques = []
                restart_deadline = min(deadline, time.time() + max(0.5, remaining / (num_restarts - r)))
                
                while time.time() < restart_deadline:
                    has_edges = False
                    for u in range(n):
                        if adj[u]:
                            has_edges = True
                            break
                    if not has_edges:
                        break
                    
                    # Randomized: pick random edge, expand
                    edges = []
                    for u in range(n):
                        for v in iter_bits(adj[u]):
                            if v > u:
                                edges.append((u, v))
                    if not edges:
                        break
                    
                    # With some probability pick random edge, else pick densest
                    if rng.random() < 0.5:
                        u, v = rng.choice(edges)
                    else:
                        best_pot = -1
                        for u2, v2 in edges:
                            common = (adj[u2] & adj[v2]).bit_count()
                            if common > best_pot:
                                best_pot = common
                                u, v = u2, v2
                    
                    # Expand
                    C = [u, v]
                    U = adj[u] & adj[v]
                    while U:
                        best_u = -1
                        best_score = -1
                        for cand in list(iter_bits(U)):
                            score = (adj[cand] & U).bit_count()
                            if score > best_score:
                                best_score = score
                                best_u = cand
                        if best_u == -1:
                            break
                        C.append(best_u)
                        U &= adj[best_u]
                    
                    if len(C) >= 2:
                        cliques.append(C)
                        remove_clique_edges(C, adj)
                
                if cliques and validate(cliques, adj_orig, n):
                    cliques = merge_compatible(cliques, adj_orig, n, deadline)
                    if validate(cliques, adj_orig, n) and len(cliques) < best_count:
                        best_count = len(cliques)
                        best_cliques = cliques

        if best_cliques is None:
            best_cliques = []
            for u in range(n):
                for v in iter_bits(adj_orig[u]):
                    if v > u:
                        best_cliques.append([u, v])
        
        return best_cliques

    results = []
    # Instance 1
    t1 = min(50, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))
    
    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(15, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))
    
    return results
