import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, leave 40s buffer)
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Uses bitmasks for speed.
        """
        # Precompute neighbor bitmasks
        neigh = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            neigh[u] = mask
        
        cliques = []
        all_mask = (1 << n) - 1
        
        def expand(R, P, X):
            if P == 0 and X == 0:
                cliques.append(R)
                return
            
            # Pivot selection: choose u in P | X that maximizes |P & neigh[u]|
            union = P | X
            if union == 0:
                return
            
            # Find pivot
            # We want to minimize |P & neigh[u]| to minimize iterations?
            # Standard BK with pivot iterates over P \ neigh[u].
            # So we want to maximize |P & neigh[u]| to minimize the set size.
            
            max_inter = -1
            pivot = -1
            u = union
            while u:
                # Extract lowest bit
                lsb = u & (-u)
                v = lsb.bit_length() - 1
                inter = P & neigh[v]
                count = inter.bit_count()
                if count > max_inter:
                    max_inter = count
                    pivot = v
                u ^= lsb
            
            # Iterate over P \ neigh[pivot]
            if pivot != -1:
                candidates = P & ~neigh[pivot]
            else:
                candidates = P
            
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_R = R | lsb
                new_P = P & neigh[v]
                new_X = X & neigh[v]
                
                expand(new_R, new_P, new_X)
                
                P &= ~lsb
                X |= lsb
        
        expand(0, all_mask, 0)
        
        # Convert bitmasks to sets of vertices
        result = []
        for mask in cliques:
            if mask == 0:
                continue
            verts = []
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            result.append(frozenset(verts))
        return result

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        
        # Filter cliques with size >= 2
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # Greedy Set Cover
        # We want to select a set of vertices S such that for every clique C in relevant_cliques, S ∩ C != empty.
        
        remaining = [set(c) for c in relevant_cliques]
        covered = [False] * len(remaining)
        selected = []
        
        # Precompute for each vertex, which cliques it belongs to
        # vertex_to_cliques: dict {vertex: set of clique indices}
        vertex_to_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(remaining):
            for v in c:
                vertex_to_cliques[v].add(i)
        
        while True:
            # Check if all covered
            if all(covered):
                break
            
            # Find vertex that covers the most uncovered cliques
            best_v = -1
            max_hits = 0
            
            for v in range(n):
                hits = 0
                for ci in vertex_to_cliques[v]:
                    if not covered[ci]:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_v = v
            
            if max_hits == 0:
                break # Should not happen
            
            selected.append(best_v)
            
            # Update coverage
            for ci in vertex_to_cliques[best_v]:
                covered[ci] = True
        
        # Post-processing: Try to remove redundant vertices
        # A vertex v is redundant if removing it still leaves all cliques covered.
        # Since the graph is sparse and cliques are small, we can check.
        
        # Sort selected by "importance" (number of cliques only it covers)
        # Actually, let's just try to remove any vertex that is redundant.
        
        # We need to know which cliques are covered by which selected vertices.
        # Let's rebuild the coverage info for the selected set.
        
        # For each clique, which selected vertices cover it?
        clique_coverers = [set() for _ in range(len(remaining))]
        for v in selected:
            for ci in vertex_to_cliques[v]:
                clique_coverers[ci].add(v)
        
        # Try to remove vertices
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                # Check if v is the only coverer for any clique
                is_only = False
                for ci in vertex_to_cliques[v]:
                    if len(clique_coverers[ci]) == 1:
                        is_only = True
                        break
                
                if not is_only:
                    # Remove v
                    selected.remove(v)
                    # Update clique_coverers
                    for ci in vertex_to_cliques[v]:
                        clique_coverers[ci].discard(v)
                    changed = True
        
        return selected

    def local_search(selected, remaining, n, vertex_to_cliques, time_limit):
        """
        Try to improve the solution using local search (1-move, 2-move).
        """
        start_ls = time.time()
        
        # Compute coverage counts for each clique
        # cover_count[ci] = number of selected vertices in clique ci
        cover_count = [0] * len(remaining)
        for v in selected:
            for ci in vertex_to_cliques[v]:
                cover_count[ci] += 1
        
        # If any cover_count is 0, solution is invalid (should not happen from greedy)
        
        # Try to remove vertices
        # If cover_count[ci] > 1 for all ci that v covers, v can be removed.
        
        # We will iterate.
        
        # Precompute: for each vertex, list of cliques it covers.
        # We already have vertex_to_cliques.
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_ls > time_limit:
                break
            
            for v in list(selected):
                # Check if v is redundant
                redundant = True
                for ci in vertex_to_cliques[v]:
                    if cover_count[ci] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove v
                    selected.remove(v)
                    for ci in vertex_to_cliques[v]:
                        cover_count[ci] -= 1
                    improved = True
            
            # Try 2-exchanges? Maybe too complex for now.
            # Let's try adding a vertex that allows removing two others?
            # This is a standard set cover local search.
            
            # For now, just do removals.
        
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Not enough time, return empty or best effort?
            # We must return a valid solution.
            # If we run out of time, we can't compute.
            # But the graphs are small. BK is fast. Greedy is fast.
            # Let's assume we have time.
            pass
            
        sol = solve_instance(n, p, seed)
        
        # Try to improve with local search if time permits
        if time.time() < deadline - 10:
            # Rebuild structures for local search
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant_cliques = [c for c in all_cliques if len(c) >= 2]
            
            if relevant_cliques:
                remaining = [set(c) for c in relevant_cliques]
                vertex_to_cliques = {v: set() for v in range(n)}
                for i, c in enumerate(remaining):
                    for v in c:
                        vertex_to_cliques[v].add(i)
                
                # Time limit for local search: 5 seconds per instance
                sol = local_search(sol, remaining, n, vertex_to_cliques, 5.0)
        
        results.append(sorted(sol))
        
        # Check time
        if time.time() > deadline - 2:
            # Break if we are running out of time, but we need 3 results.
            # If we break, we fill the rest with empty? No, that's invalid.
            # The loops should be fast enough.
            pass

    return results
