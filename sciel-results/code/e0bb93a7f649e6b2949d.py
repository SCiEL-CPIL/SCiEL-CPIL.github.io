import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 10  # Safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback for timeout, though unlikely
            # Just return a valid but potentially suboptimal partition
            # For simplicity, we'll just let it run, as it's very fast
            pass
            
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        cliques = []
        covered_internal = set()
        
        # 1. Process vertices in I
        for v in range(k, n):
            # Find neighbors in C
            N_C_v = set()
            for u in range(k):
                if (u, v) in edges:
                    N_C_v.add(u)
            
            if N_C_v:
                # Form clique {v} U N_C_v
                clique = [v] + sorted(N_C_v)
                cliques.append(clique)
                
                # Mark internal edges within N_C_v as covered
                N_C_list = sorted(N_C_v)
                for i in range(len(N_C_list)):
                    for j in range(i + 1, len(N_C_list)):
                        u1 = N_C_list[i]
                        u2 = N_C_list[j]
                        covered_internal.add((u1, u2))
        
        # 2. Process remaining internal edges
        # Build graph of uncovered internal edges
        uncovered_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) in edges:
                    if (u, w) not in covered_internal:
                        uncovered_edges.add((u, w))
        
        # Find clique partition of uncovered_edges
        # Since it's a subgraph of a clique, we can use a greedy approach
        # Find a maximal clique, remove its edges, repeat
        while uncovered_edges:
            # Find vertices involved
            vertices_involved = set()
            for u, w in uncovered_edges:
                vertices_involved.add(u)
                vertices_involved.add(w)
            
            # Greedy maximal clique
            # Sort by degree in uncovered_edges
            deg = {v: 0 for v in vertices_involved}
            for u, w in uncovered_edges:
                deg[u] += 1
                deg[w] += 1
            
            sorted_vertices = sorted(vertices_involved, key=lambda x: -deg[x])
            
            S = set()
            for v in sorted_vertices:
                # Check if v can be added to S
                ok = True
                for u in S:
                    e = (min(u, v), max(u, v))
                    if e not in uncovered_edges:
                        ok = False
                        break
                if ok:
                    S.add(v)
            
            if S:
                cliques.append(sorted(S))
                # Remove edges within S
                S_list = sorted(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        u1 = S_list[i]
                        u2 = S_list[j]
                        uncovered_edges.discard((u1, u2))
            else:
                # Should not happen if uncovered_edges is not empty
                break

        results.append(cliques)
        
    return results
