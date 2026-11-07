import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        # Reconstruct edges
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
                    
        cliques = []
        
        # 1. Handle edges in C and I
        # Strategy: For each I-vertex v, find the largest subset S_v of N(v)∩C
        # such that all pairs in S_v are not yet covered.
        
        covered_edges_in_C = set()
        
        # Sort I vertices by degree descending to fill large cliques first
        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda v: len(neighbors[v] & set(range(k))), reverse=True)
        
        for v in I_vertices:
            N_v = neighbors[v] & set(range(k))
            if not N_v:
                continue
            
            # Greedily find the largest valid subset of N_v
            # Start with all of N_v, remove vertices that cause conflicts
            S = set(N_v)
            changed = True
            while changed:
                changed = False
                for u in list(S):
                    # Check if u conflicts with any other vertex in S
                    for w in S:
                        if u < w:
                            if (u, w) in covered_edges_in_C:
                                # Remove u or w - remove the one with fewer other connections in S
                                # For simplicity, remove u
                                S.discard(u)
                                changed = True
                                break
            
            if len(S) >= 2:
                cliques.append([v] + list(S))
                for u1, u2 in combinations(sorted(S), 2):
                    covered_edges_in_C.add((u1, u2))
            elif len(S) == 1:
                u = next(iter(S))
                cliques.append([v, u])
            
            # Handle remaining neighbors of v that are not in S
            for u in N_v:
                if u not in S:
                    cliques.append([v, u])
        
        # 2. Handle remaining edges in C
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges_in_C:
                    remaining_edges.add((u, w))
        
        # Greedy clique cover for remaining_edges
        rem_set = set(remaining_edges)
        while rem_set:
            # Pick an edge, grow it into a maximal clique in the subgraph induced by rem_set
            u, w = next(iter(rem_set))
            clique_vertices = {u, w}
            
            # Try to add other vertices that connect to all in clique_vertices
            candidates = []
            for x in range(k):
                if x not in clique_vertices:
                    # Check if x connects to all in clique_vertices in rem_set
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in rem_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)
                        
            # Add all candidates
            clique_vertices.update(candidates)
            
            # Remove all edges within clique_vertices from rem_set
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                rem_set.discard((u1, u2))
            
            cliques.append(list(clique_vertices))
            
        # Convert to list of lists of ints
        results.append(cliques)
        
        if time.time() - start_time > 100:
            break
            
    return results
