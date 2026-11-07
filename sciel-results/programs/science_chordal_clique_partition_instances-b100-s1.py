"""Clique partitions of split graphs: structural greedy coloring + local refinement."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve(n, k, p, seed):
        # Reconstruct graph
        rng = random.Random(seed)
        neighbors_I = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors_I[v].add(u)
        
        # All edges in C
        all_C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_C_edges.add((u, w))
        
        def greedy_color_partition(vertices, edge_set, deadline):
            """
            Partition vertices into minimum number of independent sets in the graph (V, edge_set).
            Returns list of sets (color classes).
            Uses greedy coloring with a specific ordering.
            """
            if not vertices:
                return []
            
            # Order vertices by degree in the induced subgraph (descending)
            deg = {v: 0 for v in vertices}
            for v in vertices:
                for w in vertices:
                    if v != w and (min(v, w), max(v, w)) in edge_set:
                        deg[v] += 1
            
            ordered = sorted(vertices, key=lambda x: (-deg[x], x))
            
            color_map = {}
            used_colors = defaultdict(set) # color -> set of vertices with that color
            
            for v in ordered:
                # Find colors used by neighbors of v in the conflict graph
                forbidden = set()
                for w in vertices:
                    if v != w and (min(v, w), max(v, w)) in edge_set:
                        if w in color_map:
                            forbidden.add(color_map[w])
                
                c = 0
                while c in forbidden:
                    c += 1
                color_map[v] = c
                used_colors[c].add(v)
            
            # Group by color
            classes = []
            max_c = max(color_map.values()) if color_map else -1
            for c in range(max_c + 1):
                if c in used_colors:
                    classes.append(used_colors[c])
            
            return classes

        def solve_instance(ordering_strategy="degree_desc"):
            """
            Solves the clique partition.
            """
            covered_C_edges = set()
            cliques = []
            
            # Determine order of processing I-vertices
            I_vertices = list(range(k, n))
            if ordering_strategy == "degree_desc":
                # Process vertices with larger neighborhoods first? Or smaller?
                # In split graphs, N(v) is a subset of C.
                # If we process large N(v) first, we might cover more C-edges early.
                # However, covering C-edges early might restrict later choices.
                # Let's try sorting by size of N(v) descending first.
                I_vertices = sorted(I_vertices, key=lambda v: len(neighbors_I[v]), reverse=True)
            elif ordering_strategy == "degree_asc":
                I_vertices = sorted(I_vertices, key=lambda v: len(neighbors_I[v]))
            elif ordering_strategy == "random":
                rng_sort = random.Random(seed + 42)
                rng_sort.shuffle(I_vertices)
            
            for v in I_vertices:
                N_v = neighbors_I[v]
                if not N_v:
                    continue
                
                # The conflict graph on N_v is defined by edges in covered_C_edges.
                # We want to partition N_v into independent sets in this conflict graph.
                # An independent set in the conflict graph is a clique in the remaining C-graph.
                # So we are looking for a clique cover of N_v using cliques from the remaining C-graph.
                
                # Extract the subgraph of covered_C_edges induced by N_v
                # Wait, the conflict graph edge (a,b) exists if (a,b) is in covered_C_edges.
                # So we need to partition N_v such that no two vertices in the same part are connected in the conflict graph.
                # i.e., no two vertices in the same part have an edge in covered_C_edges.
                # This means each part must be a clique in the graph (N_v, all_C_edges - covered_C_edges).
                
                # Let G_rem be the graph on C with edges all_C_edges - covered_C_edges.
                # We need to partition N_v into cliques of G_rem.
                # This is equivalent to coloring the graph (N_v, covered_C_edges | N_v).
                
                conflict_edges = set()
                for u in N_v:
                    for w in N_v:
                        if u < w:
                            if (u, w) in covered_C_edges:
                                conflict_edges.add((u, w))
                
                # Greedy coloring of (N_v, conflict_edges)
                # We want to minimize the number of colors (cliques).
                # Standard greedy coloring order: descending degree in conflict graph.
                
                deg_conf = {u: 0 for u in N_v}
                for u in N_v:
                    for w in N_v:
                        if u != w and (min(u, w), max(u, w)) in conflict_edges:
                            deg_conf[u] += 1
                
                ordered_N_v = sorted(N_v, key=lambda x: (-deg_conf[x], x))
                
                color_map = {}
                for u in ordered_N_v:
                    forbidden = set()
                    for w in N_v:
                        if u != w and (min(u, w), max(u, w)) in conflict_edges:
                            if w in color_map:
                                forbidden.add(color_map[w])
                    
                    c = 0
                    while c in forbidden:
                        c += 1
                    color_map[u] = c
                
                # Group by color
                color_classes = defaultdict(list)
                for u, c in color_map.items():
                    color_classes[c].append(u)
                
                # For each color class, create a clique
                # Note: We must ensure the clique is valid in the ORIGINAL graph.
                # Since the color class is an independent set in the conflict graph,
                # it means no pair in the class is in covered_C_edges.
                # Since all pairs in C are edges in the original graph, and we haven't covered them yet,
                # they are available.
                # So each color class + {v} forms a valid clique.
                
                for c, S in color_classes.items():
                    if not S:
                        continue
                    clique = [v] + sorted(S)
                    cliques.append(clique)
                    
                    # Mark C-edges within S as covered
                    for u1, u2 in combinations(sorted(S), 2):
                        covered_C_edges.add((u1, u2))
            
            # Cover remaining C-edges
            remaining_C_edges = all_C_edges - covered_C_edges
            
            # Greedy clique cover for remaining_C_edges
            # We need to partition the edges of G_rem (induced by remaining_C_edges) into cliques.
            # Since G_rem is a subgraph of a complete graph (it's the complement of the union of cliques we formed),
            # it's just a graph.
            # We can use a simple greedy algorithm:
            # While remaining_C_edges is not empty:
            #   Pick an edge (u, w)
            #   Find the maximal clique containing (u, w) in G_rem
            #   Add it to cliques
            #   Remove its edges from remaining_C_edges
            
            # To speed this up, we can maintain the adjacency of G_rem.
            # But k is small (<=30), so brute force is fine.
            
            while remaining_C_edges:
                # Pick the first edge (deterministic)
                u, w = min(remaining_C_edges)
                
                # Build maximal clique containing u, w
                clique_set = {u, w}
                
                # Candidates are vertices in C
                # We can just iterate through all vertices
                for x in range(k):
                    if x in clique_set:
                        continue
                    can_add = True
                    for y in clique_set:
                        if (min(x, y), max(x, y)) not in remaining_C_edges:
                            can_add = False
                            break
                    if can_add:
                        clique_set.add(x)
                
                clique_list = sorted(clique_set)
                cliques.append(clique_list)
                
                for u1, u2 in combinations(clique_list, 2):
                    remaining_C_edges.discard((u1, u2))
            
            return cliques

        # Try multiple strategies and keep the best
        best_cliques = None
        best_count = float('inf')
        
        strategies = ["degree_desc", "degree_asc", "random"]
        
        for strat in strategies:
            try:
                cliques = solve_instance(strat)
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques
            except:
                continue
        
        if best_cliques is None:
            # Fallback: trivial cover
            cliques = []
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            for v in range(k, n):
                for u in neighbors_I[v]:
                    cliques.append([u, v])
            best_cliques = cliques
        
        return best_cliques

    result = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        cliques = solve(*spec)
        result.append(cliques)

    # Ensure we have exactly 3 results
    while len(result) < 3:
        spec = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(result)]
        cliques = []
        n, k, p, seed = spec
        rng = random.Random(seed)
        for u in range(k):
            for w in range(u + 1, k):
                cliques.append([u, w])
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cliques.append([u, v])
        result.append(cliques)

    return result[:3]
# EVOLVE-BLOCK-END
