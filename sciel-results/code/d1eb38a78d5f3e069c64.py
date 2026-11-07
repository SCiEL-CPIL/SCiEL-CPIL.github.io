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

    results = []
    start_time = time.time()

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        
        # Reconstruct graph
        # C = 0..k-1, I = k..n-1
        # Edges in C: all pairs
        # Edges in I: none
        # Cross edges: random
        
        # We need to cover all edges exactly once.
        
        # 1. Handle cross edges and some C-edges using I-vertices
        # H: set of C-edges already covered by I-cliques
        covered_C_edges = set()
        cliques = []
        
        # Process each v in I
        # To ensure determinism and speed, we iterate in order
        # We need N(v) for each v in I
        # We can regenerate the random numbers to get N(v)
        # But we need to be careful with the order of random calls in instance()
        # instance() generates:
        # 1. Edges in C (deterministic, no random)
        # 2. Cross edges: for v in k..n-1, for u in 0..k-1, if rng.random() < p
        
        # So we can just re-run the generation logic to get neighbors
        rng = random.Random(seed)
        # Skip C edges (no random)
        # Generate cross edges
        neighbors_I = [set() for _ in range(n)] # Only for I vertices
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors_I[v].add(u)
                    # neighbors_I[u].add(v) not needed as u is in C
        
        # Now process each v in I
        # We need to color the conflict graph H induced on N(v)
        # H changes as we go
        
        for v in range(k, n):
            N_v = neighbors_I[v]
            if not N_v:
                continue
                
            # Build conflict graph on N_v based on covered_C_edges
            # Vertices are elements of N_v
            # Edge (a, b) in conflict graph if (min(a,b), max(a,b)) in covered_C_edges
            
            # Compute degree in conflict graph for ordering
            deg = {}
            for u in N_v:
                deg[u] = 0
            for u in N_v:
                for w in N_v:
                    if u == w:
                        continue
                    edge = (min(u, w), max(u, w))
                    if edge in covered_C_edges:
                        deg[u] += 1
            
            # Order vertices by degree descending (ties broken by vertex id for determinism)
            ordered_N_v = sorted(N_v, key=lambda x: (-deg[x], x))
            
            # Greedy Coloring
            color_map = {}
            current_color = 0
            
            for u in ordered_N_v:
                # Find colors used by neighbors of u in the conflict graph
                used_colors = set()
                for w in ordered_N_v:
                    if u == w:
                        continue
                    edge = (min(u, w), max(u, w))
                    if edge in covered_C_edges:
                        if w in color_map:
                            used_colors.add(color_map[w])
                
                # Find smallest available color
                c = 0
                while c in used_colors:
                    c += 1
                color_map[u] = c
                if c > current_color:
                    current_color = c
            
            # Group vertices by color
            color_classes = {}
            for u, c in color_map.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
            
            # For each color class, create a clique
            for c in sorted(color_classes.keys()):
                S = color_classes[c]
                if not S:
                    continue
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Mark C-edges within S as covered
                for u1, u2 in combinations(sorted(S), 2):
                    covered_C_edges.add((u1, u2))
                    
        # 2. Handle remaining C-edges
        # Remaining edges are all pairs in C not in covered_C_edges
        remaining_C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_C_edges:
                    remaining_C_edges.add((u, w))
        
        # Greedy cover remaining_C_edges with cliques
        # While remaining_C_edges is not empty:
        #   Pick an edge (u, w)
        #   Expand to maximal clique K in the graph defined by remaining_C_edges
        #   Add K to cliques
        #   Remove edges in K from remaining_C_edges
        
        while remaining_C_edges:
            if time.time() - start_time > 110:
                # Emergency break? But we must return valid result.
                # If we break here, we might have uncovered edges.
                # The task says "Any broken rule makes the whole result invalid".
                # So we MUST finish.
                # Given k<=30, this loop is very fast.
                pass
                
            # Pick an edge
            # To be deterministic, pick the smallest edge
            u, w = min(remaining_C_edges)
            
            # Build maximal clique containing u, w
            # Candidates are vertices in C connected to all in current clique
            # Start with {u, w}
            clique_set = {u, w}
            
            # Try to add other vertices
            # Iterate through all vertices in C
            for x in range(k):
                if x in clique_set:
                    continue
                # Check if x is connected to all in clique_set in remaining_C_edges
                can_add = True
                for y in clique_set:
                    edge = (min(x, y), max(x, y))
                    if edge not in remaining_C_edges:
                        can_add = False
                        break
                if can_add:
                    clique_set.add(x)
            
            # Convert to list
            clique_list = sorted(clique_set)
            cliques.append(clique_list)
            
            # Remove edges in clique_set from remaining_C_edges
            for u1, u2 in combinations(clique_list, 2):
                remaining_C_edges.discard((u1, u2))
                
        results.append(cliques)
        
    return results
