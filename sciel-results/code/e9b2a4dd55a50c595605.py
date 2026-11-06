# EVOLVE-BLOCK-START
import time as _time
import sys

def construct():
    from random import Random
    
    start_time = _time.time()
    DEADLINE = 135  # seconds
    
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
                    # Count bits
                    cnt = bin(R_mask).count('1')
                    if cnt >= 2:
                        # Convert to frozenset
                        s = frozenset(i for i in range(n) if (R_mask >> i) & 1)
                        cliques.append(s)
                return
            
            # Choose pivot
            PX = P_mask | X_mask
            if PX == 0:
                return
            # Find pivot u in PX that maximizes |P ∩ adj[u]|
            best_u = -1
            best_count = -1
            # Iterate over bits in PX
            m = PX
            while m:
                # Get lowest set bit
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
            
            # Candidates: P \ N(best_u)
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
    
    def solve_transversal(n, cliques):
        """Find minimum transversal of cliques (each clique size >= 2)."""
        if not cliques:
            return set()
        
        clique_list = list(cliques)
        m = len(clique_list)
        
        # Map vertex -> list of clique indices
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(clique_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Greedy set cover
        uncovered = [True] * m
        uncovered_count = m
        selected = set()
        
        cov_count = [0] * n
        for v in range(n):
            for ci in vertex_to_cliques[v]:
                if uncovered[ci]:
                    cov_count[v] += 1
        
        while uncovered_count > 0:
            # Pick vertex with max coverage among uncovered
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
            # Update cov_count
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
        
        # Swap search (time-bounded)
        unselected = set(range(n)) - selected
        if unselected and selected:
            # Build set of cliques each unselected vertex covers
            # For efficiency, precompute
            time_limit = start_time + DEADLINE - 5
            for _ in range(200):  # bounded iterations
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
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if _time.time() - start_time > DEADLINE:
            while len(results) < 3:
                results.append([])
            return [sorted(r) for r in results]
        
        edges = generate_edges(n, p, seed)
        adj = build_adj_mask(n, edges)
        cliques = bron_kerbosch_bitmask(adj, n)
        transversal = solve_transversal(n, cliques)
        results.append(sorted(transversal))
    
    return [sorted(r) for r in results]
# EVOLVE-BLOCK-END
