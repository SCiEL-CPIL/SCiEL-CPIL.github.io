import time
import sys
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def dsatur_color(adj_dict, vertices):
        """DSATUR greedy coloring. Returns dict vertex -> color."""
        if not vertices:
            return {}
        colors = {}
        color_set = {u: set() for u in vertices}
        uncolored = set(vertices)
        sat_degree = {u: 0 for u in vertices}
        
        while uncolored:
            # Pick vertex with highest saturation degree, then highest degree
            best = max(uncolored, key=lambda x: (sat_degree[x], len(adj_dict.get(x, set()))))
            used_colors = color_set[best]
            c = 0
            while c in used_colors:
                c += 1
            colors[best] = c
            color_set[best].add(c)
            uncolored.discard(best)
            for w in adj_dict.get(best, set()):
                if w in uncolored:
                    if c not in color_set[w]:
                        color_set[w].add(c)
                        sat_degree[w] = len(color_set[w])
        return colors

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Internal edges of C
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        
        # Used edges in C
        used_edges = set()
        
        partition = []
        
        # Group independent vertices by their neighborhood in C
        neighborhood_groups = defaultdict(list)
        for v in I:
            N_v = frozenset(adj[v] & C_set)
            if N_v:
                neighborhood_groups[N_v].append(v)
        
        # Sort groups by neighborhood size (ascending) to process smaller ones first
        # This leaves more C-edges available for larger neighborhoods
        sorted_groups = sorted(neighborhood_groups.items(), key=lambda item: len(item[0]))
        
        # Process groups: for each unique neighborhood, handle all vertices at once
        # The shadow of a group is its neighborhood N_v.
        # Cross edges from group to N_v are covered by coloring the subgraph of C induced by N_v
        # (using only unused edges), creating one clique per color class + all group vertices.
        for N_v_frozen, group_vertices in sorted_groups:
            N_v = set(N_v_frozen)
            n_v = len(N_v)
            
            if n_v == 0:
                continue
                
            if n_v == 1:
                u = next(iter(N_v))
                for v in group_vertices:
                    partition.append(sorted([v, u]))
                continue
                
            # Build graph H on N_v using only UNUSED edges
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i+1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e in used_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
                        
            # DSATUR coloring of H (the graph of unused edges within N_v)
            # Each color class S: S is a clique in the complement of H, i.e., all edges in S are USED
            # Wait - we need S to be a clique in the remaining graph (unused edges)
            # Actually: we want to cover cross edges (v,u) for v in group, u in N_v
            # Each clique in partition that contains group vertices + some subset S of N_v
            # must have S be a clique in the graph of UNUSED edges (since those edges will be marked used)
            # So we need to partition N_v into cliques in the unused-edge graph
            # This is equivalent to coloring the complement of the unused-edge graph
            # i.e., coloring the used-edge graph within N_v
            
            # Complement of H: edge (u,w) exists if (u,w) is NOT in used_edges
            comp_adj = {u: set() for u in h_vertices}
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i+1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e not in used_edges:
                        comp_adj[u].add(w)
                        comp_adj[w].add(u)
                        
            # Color comp_adj to get clique cover of the unused-edge graph on N_v
            colors = dsatur_color(comp_adj, h_vertices)
            
            color_count = max(colors.values()) + 1 if colors else 0
            
            # Group vertices by color
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            # For each color class, create one clique containing all group vertices + color class vertices
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                    
                clique = S + group_vertices
                partition.append(sorted(clique))
                
                # Mark edges in S as used (they are unused edges that we're now covering)
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C
        remaining_edges = internal_edges - used_edges
        
        # Partition remaining_edges into cliques using greedy maximal clique extraction
        rem_set = set(remaining_edges)
        
        while rem_set:
            # Find maximal clique in the graph defined by rem_set
            # Use a greedy approach: pick vertex with highest degree in rem graph
            # Build degree map
            deg = defaultdict(int)
            for u, v in rem_set:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
                
            # Pick vertex with highest degree
            start_v = max(deg.keys(), key=lambda x: deg[x])
            
            # Greedy grow clique from start_v
            current_clique = {start_v}
            
            # Find all vertices connected to all in current_clique via rem edges
            changed = True
            while changed:
                changed = False
                candidates = set()
                for v in range(k):
                    if v in current_clique:
                        continue
                    # Check if v connects to all in current_clique
                    is_connected = True
                    for x in current_clique:
                        e = (min(v, x), max(v, x))
                        if e not in rem_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.add(v)
                
                if not candidates:
                    break
                    
                # Pick candidate with highest degree within candidates + current_clique
                best_c = max(candidates, key=lambda v: deg[v])
                current_clique.add(best_c)
                changed = True
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            # Remove all edges within current_clique from rem_set
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    rem_set.discard((min(a, b), max(a, b)))
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
