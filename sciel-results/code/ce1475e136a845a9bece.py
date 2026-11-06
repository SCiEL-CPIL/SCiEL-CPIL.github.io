# EVOLVE-BLOCK-START
import time as _time
import sys
from random import Random

def construct():
    start_time = _time.time()
    DEADLINE = 140  # seconds (10s safety margin)
    
    def generate_edges(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def build_adj_mask(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def bron_kerbosch_bitmask(adj, n):
        cliques = []
        full_mask = (1 << n) - 1
        
        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                cnt = bin(R_mask).count('1')
                if cnt >= 2:
                    s = frozenset(i for i in range(n) if (R_mask >> i) & 1)
                    cliques.append(s)
                return
            
            PX = P_mask | X_mask
            if PX == 0:
                return
            
            best_u = -1
            best_count = -1
            m = PX
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb
                cnt = bin(P_mask & adj[u]).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
                if m == 0:
                    break
            
            if best_u < 0:
                return
            
            candidates = P_mask & ~adj[best_u]
            m = candidates
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                
                new_R = R_mask | (1 << v)
                new_P = P_mask & adj[v]
                new_X = X_mask & adj[v]
                
                bk(new_R, new_P, new_X)
                
                P_mask &= ~(1 << v)
                X_mask |= (1 << v)
        
        bk(0, full_mask, 0)
        return cliques
    
    def solve_transversal(n, cliques, time_budget):
        if not cliques:
            return set()
        
        clique_list = list(cliques)
        m = len(clique_list)
        
        # Map vertex -> list of clique indices
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(clique_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Also map clique -> set of vertices (as bitmask for fast operations)
        clique_vertices = [0] * m
        for i, c in enumerate(clique_list):
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_vertices[i] = mask
        
        def greedy_set_cover(rnd_seed=0):
            """Greedy set cover with optional randomization."""
            rng = Random(rnd_seed)
            uncovered = [True] * m
            uncovered_count = m
            selected = set()
            
            cov_count = [0] * n
            for v in range(n):
                for ci in vertex_to_cliques[v]:
                    if uncovered[ci]:
                        cov_count[v] += 1
            
            while uncovered_count > 0:
                best_v = -1
                best_c = -1
                # Add small random perturbation to break ties
                for v in range(n):
                    if v not in selected:
                        score = cov_count[v]
                        if score > 0:
                            # Small random noise for diversity
                            if rnd_seed != 0:
                                score += rng.random() * 0.5
                            if score > best_c:
                                best_c = score
                                best_v = v
                if best_v < 0 or best_c <= 0:
                    break
                selected.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    if uncovered[ci]:
                        uncovered[ci] = False
                        uncovered_count -= 1
                for v in range(n):
                    if v not in selected:
                        new_c = 0
                        for ci in vertex_to_cliques[v]:
                            if uncovered[ci]:
                                new_c += 1
                        cov_count[v] = new_c
            
            return selected
        
        def local_search(selected, time_limit):
            """Local search: removal, 1-swap, 2-swap."""
            clique_coverage = [0] * m
            for v in selected:
                for ci in vertex_to_cliques[v]:
                    clique_coverage[ci] += 1
            
            improved = True
            while improved and _time.time() < time_limit:
                improved = False
                
                # Phase 1: Remove redundant vertices
                for v in list(selected):
                    removable = True
                    for ci in vertex_to_cliques[v]:
                        if clique_coverage[ci] <= 1:
                            removable = False
                            break
                    if removable:
                        selected.discard(v)
                        for ci in vertex_to_cliques[v]:
                            clique_coverage[ci] -= 1
                        improved = True
                        break
                
                if _time.time() >= time_limit:
                    break
                
                # Phase 2: 1-swap (replace one selected with one unselected)
                unselected = set(range(n)) - selected
                if unselected and selected:
                    found_swap = False
                    for v_sel in list(selected):
                        unique_cliques = [ci for ci in vertex_to_cliques[v_sel] if clique_coverage[ci] == 1]
                        if not unique_cliques:
                            continue
                        for v_un in unselected:
                            # Check if v_un covers all unique cliques of v_sel
                            covers_all = True
                            for ci in unique_cliques:
                                if ci not in vertex_to_cliques[v_un]:
                                    covers_all = False
                                    break
                            if not covers_all:
                                continue
                            # Check if removing v_sel doesn't leave any clique uncovered
                            ok = True
                            for ci in vertex_to_cliques[v_sel]:
                                if ci not in vertex_to_cliques[v_un] and clique_coverage[ci] <= 1:
                                    ok = False
                                    break
                            if ok:
                                selected.discard(v_sel)
                                selected.add(v_un)
                                for ci in vertex_to_cliques[v_sel]:
                                    clique_coverage[ci] -= 1
                                for ci in vertex_to_cliques[v_un]:
                                    clique_coverage[ci] += 1
                                unselected.discard(v_un)
                                unselected.add(v_sel)
                                found_swap = True
                                break
                        if found_swap:
                            break
                    if found_swap:
                        improved = True
                        continue
                
                if _time.time() >= time_limit:
                    break
                
                # Phase 3: 2-swap (replace one selected with two unselected, or two selected with one)
                # Try: remove one selected, add two unselected if it reduces size by 1
                # Actually, let's try: replace 2 selected with 1 unselected (size reduction)
                if len(selected) >= 2:
                    found_2swap = False
                    sel_list = list(selected)
                    for i in range(len(sel_list)):
                        for j in range(i + 1, len(sel_list)):
                            v1, v2 = sel_list[i], sel_list[j]
                            # Find cliques only covered by v1 or v2
                            only_v1 = [ci for ci in vertex_to_cliques[v1] if clique_coverage[ci] == 1 and ci not in vertex_to_cliques[v2]]
                            only_v2 = [ci for ci in vertex_to_cliques[v2] if clique_coverage[ci] == 1 and ci not in vertex_to_cliques[v1]]
                            if not only_v1 and not only_v2:
                                # Both are redundant in some sense, but already handled in removal
                                continue
                            # Try to find one unselected vertex that covers all of only_v1 + only_v2
                            combined = only_v1 + only_v2
                            if not combined:
                                continue
                            for v_un in unselected:
                                covers_all = all(ci in vertex_to_cliques[v_un] for ci in combined)
                                if covers_all:
                                    # Check no other cliques become uncovered
                                    ok = True
                                    for ci in vertex_to_cliques[v1]:
                                        if ci not in vertex_to_cliques[v2] and ci not in vertex_to_cliques[v_un] and clique_coverage[ci] <= 1:
                                            ok = False
                                            break
                                    if ok:
                                        for ci in vertex_to_cliques[v2]:
                                            if ci not in vertex_to_cliques[v1] and ci not in vertex_to_cliques[v_un] and clique_coverage[ci] <= 1:
                                                ok = False
                                                break
                                    if ok:
                                        selected.discard(v1)
                                        selected.discard(v2)
                                        selected.add(v_un)
                                        for ci in vertex_to_cliques[v1]:
                                            clique_coverage[ci] -= 1
                                        for ci in vertex_to_cliques[v2]:
                                            clique_coverage[ci] -= 1
                                        for ci in vertex_to_cliques[v_un]:
                                            clique_coverage[ci] += 1
                                        unselected.discard(v_un)
                                        unselected.add(v1)
                                        unselected.add(v2)
                                        found_2swap = True
                                        break
                                if found_2swap:
                                    break
                            if found_2swap:
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        improved = True
                        continue
                
                # Phase 4: Try adding one unselected vertex that covers many uncovered... 
                # (Not applicable since we need to cover ALL cliques, not minimize uncovered)
                # Instead, try: replace 1 selected with 2 unselected if it allows removing more later
                # This is complex; skip for now and rely on restarts
        
            return selected
        
        # Multi-restart strategy
        best_solution = None
        best_size = float('inf')
        
        # Determine number of restarts based on time budget
        # Each greedy + local search takes roughly O(n * m * iterations)
        # Estimate: for n=40, m~50-200, one pass is fast (~10ms)
        # So we can do many restarts
        
        elapsed = _time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 0:
            remaining = 1
        
        # Do at least 1 greedy, then use remaining time for restarts
        # Estimate time per restart
        n_restarts = max(1, int(remaining * 20))  # ~50ms per restart estimate
        n_restarts = min(n_restarts, 50)  # Cap at 50 restarts
        
        for restart in range(n_restarts):
            if _time.time() - start_time > time_budget - 2:
                break
            
            if restart == 0:
                # Deterministic greedy
                selected = greedy_set_cover(rnd_seed=0)
            else:
                # Randomized greedy
                selected = greedy_set_cover(rnd_seed=restart)
            
            # Local search
            ls_time_limit = min(_time.time() + remaining / n_restarts, time_budget - 2)
            selected = local_search(selected, ls_time_limit)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)
        
        return best_solution if best_solution is not None else set()
    
    results = []
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    
    # Allocate time: give more to larger instances
    # Instance sizes: 30, 35, 40 vertices
    # Time allocation proportional to n^2 (clique enumeration cost)
    weights = [30*30, 35*35, 40*40]
    total_w = sum(weights)
    base_time = DEADLINE - 5  # Leave 5s margin
    
    for idx, (n, p, seed) in enumerate(instances):
        if _time.time() - start_time > DEADLINE - 3:
            while len(results) < 3:
                results.append([])
            return [sorted(r) for r in results]
        
        # Time budget for this instance
        instance_budget = (base_time * weights[idx]) / total_w
        # But ensure we have enough time for remaining instances
        remaining_instances = len(instances) - idx - 1
        min_remaining = sum(weights[j] for j in range(idx + 1, len(instances))) / total_w * base_time
        instance_budget = max(instance_budget, 5)  # At least 5s
        instance_budget = min(instance_budget, base_time - min_remaining)
        
        edges = generate_edges(n, p, seed)
        adj = build_adj_mask(n, edges)
        cliques = bron_kerbosch_bitmask(adj, n)
        transversal = solve_transversal(n, cliques, DEADLINE - 3)
        results.append(sorted(transversal))
    
    return [sorted(r) for r in results]
# EVOLVE-BLOCK-END
