# EVOLVE-BLOCK-START
import time as _time
import sys

def construct():
    from random import Random
    
    start_time = _time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    def generate_edges(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def build_adj_mask(n, edges):
        """Build adjacency as list of integer bitmasks."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def bron_kerbosch_bitmask(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot, bitmask version.
        Returns list of frozensets with size >= 2."""
        cliques = []
        full_mask = (1 << n) - 1
        
        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                if R_mask != 0:
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
    
    def max_independent_set(adj_comp, n, time_limit=None):
        """Find maximum independent set in graph adj_comp (bitmask adjacency) using
        Bron-Kerbosch on complement (which is max clique in complement = max IS in original).
        Returns the independent set as a frozenset."""
        
        # Build complement adjacency: comp_adj[v] = vertices NOT adjacent to v in adj_comp
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for v in range(n):
            # In complement, v is connected to all except itself and its adj_comp neighbors
            comp_adj[v] = full_mask & ~(1 << v) & ~adj_comp[v]
        
        best_is = frozenset()
        best_size = 0
        
        def bk_clique(R_mask, P_mask, X_mask):
            nonlocal best_is, best_size
            
            if P_mask == 0 and X_mask == 0:
                size = bin(R_mask).count('1')
                if size > best_size:
                    best_size = size
                    best_is = frozenset(i for i in range(n) if (R_mask >> i) & 1)
                return
            
            # Pruning: even if we add all of P, can't beat best
            if bin(R_mask | P_mask).count('1') <= best_size:
                return
            
            PX = P_mask | X_mask
            if PX == 0:
                return
            
            # Pivot: choose u in PX maximizing |P ∩ comp_adj[u]|
            best_u = -1
            best_count = -1
            m = PX
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb
                cnt = bin(P_mask & comp_adj[u]).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
                if m == 0:
                    break
            
            if best_u < 0:
                return
            
            candidates = P_mask & ~comp_adj[best_u]
            
            m = candidates
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                
                bk_clique(R_mask | (1 << v), P_mask & comp_adj[v], X_mask & comp_adj[v])
                
                P_mask &= ~(1 << v)
                X_mask |= (1 << v)
                
                if time_limit is not None and _time.time() > time_limit:
                    return
        
        bk_clique(0, full_mask, 0)
        return best_is
    
    def solve_transversal(n, cliques, time_limit=None):
        """Find minimum transversal of cliques (each clique size >= 2) using
        maximum independent set formulation."""
        if not cliques:
            return set()
        
        # Build "co-occurrence graph": edge (u,v) if u and v appear in same maximal clique
        # The complement of this graph's vertex cover = maximum independent set
        # A vertex cover in the co-occurrence graph = set of vertices hitting every edge
        # But we need to hit every clique, not every edge...
        
        # Actually, the correct formulation:
        # We need a set S that intersects every maximal clique C.
        # This is the "hitting set" / "transversal" problem.
        # 
        # Key insight: If we define a hypergraph where hyperedges are the maximal cliques,
        # we want the minimum hitting set. This is NP-hard in general.
        #
        # The MIS formulation: Create graph H where vertices are original vertices,
        # and edge (u,v) exists if u,v co-occur in some maximal clique.
        # Then a set S is a transversal iff S is a vertex cover of H? NO - that's not right either.
        #
        # Let me reconsider. The greedy + local search approach is more direct.
        # Let me improve it with better initialization and more thorough local search.
        
        clique_list = list(cliques)
        m = len(clique_list)
        
        # Map vertex -> list of clique indices
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(clique_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Greedy set cover with tie-breaking by degree
        uncovered = [True] * m
        uncovered_count = m
        selected = set()
        
        # Precompute coverage counts
        cov_count = [0] * n
        for v in range(n):
            for ci in vertex_to_cliques[v]:
                if uncovered[ci]:
                    cov_count[v] += 1
        
        while uncovered_count > 0:
            best_v = -1
            best_c = -1
            for v in range(n):
                if v not in selected and cov_count[v] > best_c:
                    best_c = cov_count[v]
                    best_v = v
            if best_v < 0 or best_c == 0:
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
        
        # Build clique coverage
        clique_coverage = [0] * m
        for v in selected:
            for ci in vertex_to_cliques[v]:
                clique_coverage[ci] += 1
        
        # Local search: removal
        improved = True
        while improved:
            improved = False
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
        
        # 2-swap local search
        unselected = set(range(n)) - selected
        if unselected and selected and time_limit is not None:
            for _ in range(500):
                if _time.time() > time_limit:
                    break
                found_swap = False
                for v_sel in list(selected):
                    unique_cliques = [ci for ci in vertex_to_cliques[v_sel] if clique_coverage[ci] == 1]
                    if not unique_cliques:
                        continue
                    for v_un in unselected:
                        covers_all = all(ci in vertex_to_cliques[v_un] for ci in unique_cliques)
                        if covers_all:
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
                if not found_swap:
                    break
                # Re-run removal after swap
                improved = True
                while improved:
                    improved = False
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
        
        return selected
    
    def solve_transversal_mis(n, cliques, time_limit=None):
        """Solve transversal via maximum independent set on the conflict graph.
        
        Conflict graph: vertices = original vertices, edge (u,v) if u and v are in
        the same maximal clique. A set S is a transversal (hits every clique) iff
        V \\ S is an independent set in the conflict graph (no two non-selected
        vertices co-occur in any clique... wait, that's not right either).
        
        Actually: S is a transversal iff for every clique C, C ∩ S ≠ ∅.
        Equivalently, V \\ S contains no entire clique.
        This is NOT the same as V \\ S being an independent set in the conflict graph.
        
        So the MIS formulation doesn't directly work. Let me use a different approach:
        branch and bound on the hitting set.
        """
        if not cliques:
            return set()
        
        clique_list = list(cliques)
        m = len(clique_list)
        
        # Sort cliques by size (smallest first) for better branching
        clique_list.sort(key=len)
        
        # Map vertex -> list of clique indices
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(clique_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Greedy initial solution
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
            for v in range(n):
                if v not in selected and cov_count[v] > best_c:
                    best_c = cov_count[v]
                    best_v = v
            if best_v < 0 or best_c == 0:
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
        
        best_solution = set(selected)
        best_size = len(selected)
        
        # Branch and bound
        def branch_and_bound(uncovered_set, current, depth):
            nonlocal best_solution, best_size
            
            if len(current) >= best_size:
                return
            
            if not uncovered_set:
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = set(current)
                return
            
            # Pick the most constrained clique (fewest available vertices)
            # Among uncovered, pick clique with fewest vertices not in current
            best_clique = None
            best_count = n + 1
            for ci in uncovered_set:
                avail = 0
                for v in clique_list[ci]:
                    if v not in current:
                        avail += 1
                if avail < best_count:
                    best_count = avail
                    best_clique = ci
            
            if best_clique is None or best_count == 0:
                return
            
            # Branch on each vertex in the chosen clique
            for v in clique_list[best_clique]:
                if v in current:
                    continue
                new_current = current | {v}
                new_uncovered = set()
                for ci in uncovered_set:
                    if v in clique_list[ci]:
                        continue
                    new_uncovered.add(ci)
                branch_and_bound(new_uncovered, new_current, depth + 1)
        
        # Use B&B with time limit
        uncovered_all = set(range(m))
        branch_and_bound(uncovered_all, set(), 0)
        
        return best_solution
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if _time.time() - start_time > DEADLINE:
            while len(results) < 3:
                results.append([])
            return [sorted(r) for r in results]
        
        edges = generate_edges(n, p, seed)
        adj = build_adj_mask(n, edges)
        cliques = bron_kerbosch_bitmask(adj, n)
        
        # Try both approaches and take the better one
        tl = min(start_time + DEADLINE - 2, start_time + 30)
        sol1 = solve_transversal(n, cliques, time_limit=tl)
        
        tl2 = min(start_time + DEADLINE - 2, start_time + 50)
        sol2 = solve_transversal_mis(n, cliques, time_limit=tl2)
        
        best = sol1 if len(sol1) <= len(sol2) else sol2
        results.append(sorted(best))
    
    return [sorted(r) for r in results]
# EVOLVE-BLOCK-END
