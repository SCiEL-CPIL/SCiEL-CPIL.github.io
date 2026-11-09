"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # I edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)
    
    def dsatur_coloring(vertices, adj):
        """DSATUR greedy coloring. Returns dict vertex->color."""
        if not vertices:
            return {}
        colors = {}
        sat_deg = {v: 0 for v in vertices}
        color_sets = {v: set() for v in vertices}
        uncolored = set(vertices)
        
        while uncolored:
            # Pick vertex with highest saturation degree, tie-break by degree in uncolored
            best = max(uncolored, key=lambda x: (sat_deg[x], len(adj[x] & uncolored)))
            # Smallest available color
            c = 0
            while c in color_sets[best]:
                c += 1
            colors[best] = c
            color_sets[best].add(c)
            uncolored.discard(best)
            # Update neighbors
            for w in adj[best]:
                if w in uncolored and c not in color_sets[w]:
                    color_sets[w].add(c)
                    sat_deg[w] = len(color_sets[w])
        return colors
    
    def greedy_max_clique(remaining_edges, adj, candidates):
        """Find a large clique from remaining_edges using greedy expansion."""
        if not remaining_edges:
            return set()
        
        # Pick an edge to start
        u, v = next(iter(remaining_edges))
        clique = {u, v}
        
        # Candidates must be connected to all in clique via remaining edges
        common = set(candidates)
        for x in clique:
            common &= adj[x]
        common -= clique
        
        # Sort by degree in remaining graph for better greedy
        def rem_degree(x):
            return sum(1 for y in clique if (min(x, y), max(x, y)) in remaining_edges)
        
        sorted_cands = sorted(common, key=lambda x: len(adj[x] & candidates), reverse=True)
        
        changed = True
        while changed:
            changed = False
            for w in sorted_cands:
                if w in clique:
                    continue
                # Check if w connects to all in clique via remaining edges
                can_add = all((min(w, z), max(w, z)) in remaining_edges for z in clique)
                if can_add:
                    clique.add(w)
                    changed = True
                    break
        
        return clique if len(clique) >= 2 else set()
    
    def solve_instance(n, k, p, seed):
        # Build adjacency
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Generate edges
        rng = random.Random(seed)
        N = {v: set() for v in I}
        
        # C-C edges (all present)
        # C-I edges
        for v in I:
            for u in C:
                if rng.random() < p:
                    N[v].add(u)
        
        # All C-internal edges
        all_c_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_c_edges.add((u, v))
        
        used_c_edges = set()
        partition = []
        
        # Sort I vertices by |N(v)| ascending - process smaller neighborhoods first
        # This leaves larger neighborhoods for later, which may have more unused edges
        I_sorted = sorted(I, key=lambda v: len(N[v]))
        
        for v in I_sorted:
            if time.time() > deadline:
                break
            Nv = N[v]
            if not Nv:
                continue
            
            nv = len(Nv)
            if nv == 1:
                u = next(iter(Nv))
                partition.append(sorted([v, u]))
                continue
            
            # Build graph H on Nv: edge (u,w) if (u,w) is in all_c_edges and NOT in used_c_edges
            h_vertices = list(Nv)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i + 1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e in all_c_edges and e not in used_c_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
            
            # Color H using DSATUR
            colors = dsatur_coloring(h_vertices, h_adj)
            
            # Group by color
            color_classes = defaultdict(list)
            for u, c in colors.items():
                color_classes[c].append(u)
            
            for c in sorted(color_classes.keys()):
                S = sorted(color_classes[c])
                if len(S) < 1:
                    continue
                clique = sorted(S + [v])
                partition.append(clique)
                # Mark all C-edges within S as used
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = S[i], S[j]
                        used_c_edges.add((min(a, b), max(a, b)))
        
        # Cover remaining C-edges
        remaining_edges = all_c_edges - used_c_edges
        
        if remaining_edges:
            # Greedy clique cover for residual C-graph
            rem_adj = {u: set() for u in C}
            for (u, v) in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            remaining = set(remaining_edges)
            
            while remaining:
                if time.time() > deadline:
                    # Fallback: cover each edge individually
                    for (u, v) in remaining:
                        partition.append([u, v])
                    break
                
                # Find a large clique in the remaining graph
                clique = greedy_max_clique(remaining, rem_adj, set(C))
                
                if not clique or len(clique) < 2:
                    # Fallback to single edge
                    u, v = next(iter(remaining))
                    clique = {u, v}
                
                partition.append(sorted(clique))
                
                # Remove all edges within clique from remaining
                clique_list = sorted(clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        remaining.discard((min(a, b), max(a, b)))
        
        # Convert to list of lists of ints
        final_cliques = []
        for c in partition:
            final_cliques.append([int(x) for x in c])
        
        return final_cliques
    
    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Return empty lists for remaining instances
            results.append([])
            continue
        result = solve_instance(n, k, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
