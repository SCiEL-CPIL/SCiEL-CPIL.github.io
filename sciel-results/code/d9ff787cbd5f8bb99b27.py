import time
import random
import math

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def solve_instance(n, p, seed, time_budget):
        edges, adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            return []

        cliques_list = cliques
        m = len(cliques_list)
        
        # Precompute vertex to cliques mapping
        # C[v] is a set of clique indices
        C = [set() for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                C[v].add(i)
        
        def greedy_init(rng):
            uncovered = set(range(m))
            selected = set()
            
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not C[v]:
                        continue
                    count = len(uncovered & C[v])
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count and count > 0:
                        if rng.random() < 0.5:
                            best_v = v
                
                if best_v == -1:
                    break
                    
                selected.add(best_v)
                uncovered -= C[best_v]
                
            return selected

        def lagrangian_set_cover(rng, max_iters=200):
            """
            Use Lagrangian relaxation / subgradient method to guide vertex selection.
            Each clique gets a dual variable (price). Vertices are selected based on
            weighted marginal utility.
            """
            # Initialize dual variables
            dual = [1.0] * m  # price for each clique
            
            best_solution = None
            best_size = n + 1
            
            for it in range(max_iters):
                # Compute marginal utility for each vertex
                # Utility[v] = sum of dual[c] for c in C[v] - vertex_cost
                # We use a small vertex cost to encourage sparsity
                vertex_cost = 0.1
                
                utilities = []
                for v in range(n):
                    util = sum(dual[c] for c in C[v]) - vertex_cost
                    utilities.append(util)
                
                # Greedy selection based on utilities
                uncovered = set(range(m))
                selected = set()
                
                while uncovered:
                    best_v = -1
                    best_util = -1
                    for v in range(n):
                        if not C[v]:
                            continue
                        # Only consider vertices that cover at least one uncovered clique
                        cov_count = len(uncovered & C[v])
                        if cov_count == 0:
                            continue
                        # Weighted utility: sum of duals for uncovered cliques covered
                        util = sum(dual[c] for c in C[v] if c in uncovered) - vertex_cost * 0.5
                        if util > best_util:
                            best_util = util
                            best_v = v
                    
                    if best_v == -1:
                        break
                    
                    selected.add(best_v)
                    uncovered -= C[best_v]
                
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                
                # Update dual variables using subgradient method
                # subgradient[c] = 1 - cover_count[c] (positive if uncovered)
                cover_count = [0] * m
                for v in selected:
                    for c in C[v]:
                        cover_count[c] += 1
                
                # Subgradient: if clique is uncovered, increase its price
                # If over-covered, decrease
                step_size = 0.5 / (it + 1)
                for c in range(m):
                    if cover_count[c] == 0:
                        dual[c] += step_size
                    elif cover_count[c] > 1:
                        dual[c] = max(0.1, dual[c] - step_size * 0.5)
                    else:
                        dual[c] = max(0.1, dual[c] - step_size * 0.1)
            
            return best_solution if best_solution else set(range(n))

        def local_search(S, rng, time_limit):
            S = set(S)
            cover_count = [0] * m
            for v in S:
                for c in C[v]:
                    cover_count[c] += 1
            
            improved = True
            start = time.time()
            
            while improved and (time.time() - start) < time_limit:
                improved = False
                
                # 1. Try Removals
                for v in list(S):
                    can_remove = True
                    for c in C[v]:
                        if cover_count[c] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for c in C[v]:
                            cover_count[c] -= 1
                        S.remove(v)
                        improved = True
                        break
                
                if improved:
                    continue
                    
                # 2. Try 1-for-1 Swaps
                found_swap = False
                for v_out in list(S):
                    critical = [c for c in C[v_out] if cover_count[c] == 1]
                    if not critical:
                        continue
                    
                    best_v_in = -1
                    best_score = -1
                    
                    for v_in in range(n):
                        if v_in in S:
                            continue
                        valid = True
                        for c in critical:
                            if c not in C[v_in]:
                                valid = False
                                break
                        if valid:
                            score = len(C[v_in])
                            if score > best_score:
                                best_score = score
                                best_v_in = v_in
                                if score == m:
                                    break
                    
                    if best_v_in != -1:
                        for c in C[v_out]:
                            cover_count[c] -= 1
                        for c in C[best_v_in]:
                            cover_count[c] += 1
                        
                        S.remove(v_out)
                        S.add(best_v_in)
                        found_swap = True
                        break
                
                if found_swap:
                    improved = True
                    continue
                    
                # 3. Try 2-for-1 swaps: remove two vertices, add one
                # This can help when no single swap works but two removals enable a better addition
                found_2swap = False
                S_list = list(S)
                for i in range(len(S_list)):
                    if (time.time() - start) >= time_limit:
                        break
                    for j in range(i + 1, len(S_list)):
                        v1 = S_list[i]
                        v2 = S_list[j]
                        # Find all cliques that would become uncovered
                        # A clique c is critical if cover_count[c] == 1 and c in C[v1] or C[v2]
                        # Actually, we need: after removing v1 and v2, which cliques are uncovered?
                        # A clique c is uncovered after removal if:
                        #   cover_count[c] == 1 and (c in C[v1] or c in C[v2])
                        #   cover_count[c] == 2 and c in C[v1] and c in C[v2]
                        critical_cliques = set()
                        for c in C[v1]:
                            if cover_count[c] == 1:
                                critical_cliques.add(c)
                            elif cover_count[c] == 2 and c in C[v2]:
                                critical_cliques.add(c)
                        for c in C[v2]:
                            if cover_count[c] == 1 and c not in C[v1]:
                                critical_cliques.add(c)
                        
                        if not critical_cliques:
                            # Both vertices are redundant together
                            for c in C[v1]:
                                cover_count[c] -= 1
                            for c in C[v2]:
                                cover_count[c] -= 1
                            S.remove(v1)
                            S.remove(v2)
                            found_2swap = True
                            break
                        
                        # Find a vertex that covers all critical_cliques
                        best_v_in = -1
                        best_score = -1
                        for v_in in range(n):
                            if v_in in S:
                                continue
                            valid = True
                            for c in critical_cliques:
                                if c not in C[v_in]:
                                    valid = False
                                    break
                            if valid:
                                score = len(C[v_in])
                                if score > best_score:
                                    best_score = score
                                    best_v_in = v_in
                                    if score == m:
                                        break
                        
                        if best_v_in != -1:
                            # Perform the 2-for-1 swap
                            for c in C[v1]:
                                cover_count[c] -= 1
                            for c in C[v2]:
                                cover_count[c] -= 1
                            for c in C[best_v_in]:
                                cover_count[c] += 1
                            
                            S.remove(v1)
                            S.remove(v2)
                            S.add(best_v_in)
                            found_2swap = True
                            break
                    
                    if found_2swap:
                        break
                
                if found_2swap:
                    improved = True

            return S

        best_set = set(range(n)) # Fallback
        start_time = time.time()
        
        # Run multiple iterations with different strategies
        iter_count = 0
        while (time.time() - start_time) < time_budget:
            iter_seed = 1000 + iter_count
            rng = random.Random(iter_seed)
            
            # Alternate between greedy and Lagrangian approaches
            if iter_count % 3 == 0:
                # Pure greedy
                candidate = greedy_init(rng)
            elif iter_count % 3 == 1:
                # Lagrangian-guided
                candidate = lagrangian_set_cover(rng, max_iters=50)
            else:
                # Greedy with random perturbation
                candidate = greedy_init(rng)
                # Randomly remove some vertices and re-add
                if len(candidate) > 2:
                    to_remove = rng.sample(sorted(candidate), min(2, len(candidate) // 2))
                    for v in to_remove:
                        candidate.discard(v)
                    # Re-greedy to cover remaining
                    uncovered = set()
                    for c in range(m):
                        covered = False
                        for v in candidate:
                            if c in C[v]:
                                covered = True
                                break
                        if not covered:
                            uncovered.add(c)
                    while uncovered:
                        best_v = -1
                        best_count = -1
                        for v in range(n):
                            if v in candidate or not C[v]:
                                continue
                            count = len(uncovered & C[v])
                            if count > best_count:
                                best_count = count
                                best_v = v
                        if best_v == -1:
                            break
                        candidate.add(best_v)
                        uncovered -= C[best_v]
            
            # Local Search
            remaining_ls = time_budget - (time.time() - start_time)
            ls_time = min(0.1, remaining_ls * 0.3)
            if ls_time > 0.01:
                candidate = local_search(candidate, rng, ls_time)
            
            if len(candidate) < len(best_set):
                best_set = candidate
                
            iter_count += 1
            
        return sorted(best_set)

    out = []
    start_time = time.time()
    total_deadline = 115.0 # Leave 5s margin
    
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining = total_deadline - (time.time() - start_time)
        if remaining <= 0:
            out.append(sorted(range(n)))
            continue
        
        # Allocate time proportionally
        # Simple: divide remaining by instances left
        inst_budget = remaining / (len(instances) - idx)
        
        res = solve_instance(n, p, seed, inst_budget)
        out.append(res)
        
    return out
