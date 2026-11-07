import time
import random
from itertools import combinations

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
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        # Build list of I vertices
        I_vertices = list(range(k, n))
        
        # For each I vertex, N(v) is neighbors in C
        N_v = {v: neighbors[v] & set(range(k)) for v in I_vertices}
        
        # Initialize partitions for each I vertex
        # parts[v] is a list of sets, each set is a subset of N(v)
        # Initially, one part: N(v)
        parts = {}
        for v in I_vertices:
            if N_v[v]:
                parts[v] = [set(N_v[v])]
            else:
                parts[v] = []
        
        # Function to check conflicts and count covered edges
        def get_covered_cc_edges():
            covered = set()
            for v in I_vertices:
                for part in parts[v]:
                    for u, w in combinations(sorted(part), 2):
                        covered.add((min(u, w), max(u, w)))
            return covered

        # Iteratively resolve conflicts
        # A conflict is a C-C edge covered by more than 1 I-vertex
        # We track how many I-vertices cover each C-C edge
        # And which part they are in
        
        max_iterations = 1000
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Count coverage of each C-C edge by I-vertices
            edge_cover_count = {} # edge -> list of (v, part_index)
            for v in I_vertices:
                for idx, part in enumerate(parts[v]):
                    for u, w in combinations(sorted(part), 2):
                        e = (min(u, w), max(u, w))
                        if e not in edge_cover_count:
                            edge_cover_count[e] = []
                        edge_cover_count[e].append((v, idx))
            
            # Find conflicts
            conflicts = []
            for e, covers in edge_cover_count.items():
                if len(covers) > 1:
                    conflicts.append((e, covers))
            
            if not conflicts:
                break
            
            # Pick a conflict to resolve
            # Resolve by splitting a part
            # Choose the conflict with the most covers? Or just the first one?
            # Let's pick the first one
            e, covers = conflicts[0]
            u, w = e
            
            # We need to ensure u and w are not in the same part for at least one of the v's
            # We can split the part for the first v in covers
            v, idx = covers[0]
            part = parts[v][idx]
            
            # Move u to a new part
            # Create new part with {u}
            new_part = {u}
            part.remove(u)
            parts[v].append(new_part)
            
            # Note: This might create new conflicts or resolve them.
            # We loop again to check.
            
        # Now, determine the final cliques
        cliques = []
        
        # 1. Add I-cliques
        # For each v, for each part, if part has size >= 1, add {v} U part
        # But the clique must have at least 2 vertices.
        # If part is empty, skip.
        # If part has 1 vertex u, clique is {v, u}. Size 2. OK.
        
        covered_cc = set()
        for v in I_vertices:
            for part in parts[v]:
                if not part:
                    continue
                clique = [v] + sorted(part)
                cliques.append(clique)
                for u, w in combinations(sorted(part), 2):
                    covered_cc.add((min(u, w), max(u, w)))
                    
        # 2. Cover remaining C-C edges
        # Remaining edges are all C-C edges not in covered_cc
        # C-C edges are all pairs in C
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                e = (u, w)
                if e not in covered_cc:
                    remaining_edges.add(e)
                    
        # Partition remaining_edges into cliques
        # Greedy: pick an edge, find maximal clique in remaining graph
        while remaining_edges:
            if time.time() > deadline:
                break
            u, w = next(iter(remaining_edges))
            clique = {u, w}
            
            # Try to extend clique
            # A vertex x can be added if (x, y) is in remaining_edges for all y in clique
            # And x is in C (0 to k-1)
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x in clique:
                        continue
                    can_add = True
                    for y in clique:
                        e = (min(x, y), max(x, y))
                        if e not in remaining_edges:
                            can_add = False
                            break
                    if can_add:
                        clique.add(x)
                        changed = True
            
            # Add clique
            cliques.append(sorted(clique))
            
            # Remove edges
            for u1, u2 in combinations(sorted(clique), 2):
                remaining_edges.discard((min(u1, u2), max(u1, u2)))
                
        # Final check and cleanup
        final_cliques = []
        for cl in cliques:
            # Ensure distinct ints
            cl = list(set(cl))
            if len(cl) >= 2:
                final_cliques.append([int(x) for x in sorted(cl)])
        
        results.append(final_cliques)
    
    return results
