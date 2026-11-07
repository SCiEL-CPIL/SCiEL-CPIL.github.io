import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10s safety margin, plenty of time
    
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
        return sorted(edges)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return empty or seed-like? 
            # Better to just break and return what we have? 
            # But we need 3 results. 
            # If we timeout, we should probably just use a very fast fallback for remaining specs.
            # For now, assume we don't timeout.
            pass
            
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Separate internal and cross edges
        internal_edges = set()
        cross_edges = {} # v -> set of u
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((u, v))
            else:
                # One is in C, one in I
                if u < k:
                    cross_edges.setdefault(v, set()).add(u)
                else:
                    cross_edges.setdefault(u, set()).add(v)
                    
        uncovered_internal = set(internal_edges)
        uncovered_cross = {v: set(us) for v, us in cross_edges.items() if us}
        
        cliques = []
        
        # Helper to find a clique in a graph represented by an edge set
        # Graph vertices are implicitly those in 'candidates'
        # Edges must be in 'edge_set'
        def find_clique(candidates, edge_set, max_size=None):
            if not candidates:
                return set()
            
            # Sort candidates by degree in the subgraph defined by edge_set
            # This helps greedy algorithms find larger cliques
            deg = {v: 0 for v in candidates}
            for u in candidates:
                for w in candidates:
                    if u == w: continue
                    e = (min(u, w), max(u, w))
                    if e in edge_set:
                        deg[u] += 1
            
            sorted_candidates = sorted(candidates, key=lambda x: -deg[x])
            
            S = set()
            for u in sorted_candidates:
                # Check if u can be added to S
                ok = True
                for w in S:
                    e = (min(u, w), max(u, w))
                    if e not in edge_set:
                        ok = False
                        break
                if ok:
                    S.add(u)
                    if max_size and len(S) >= max_size:
                        break
            
            return S

        # Phase 1: Cover cross edges with mixed cliques
        # We process vertices in I in order of decreasing uncovered degree
        while uncovered_cross:
            # Pick v with max degree
            max_v = None
            max_deg = -1
            for v, us in uncovered_cross.items():
                if len(us) > max_deg:
                    max_deg = len(us)
                    max_v = v
            
            if max_v is None:
                break
                
            U_v = uncovered_cross[max_v]
            
            # Find a clique S in U_v such that all edges in S are in uncovered_internal
            S = find_clique(U_v, uncovered_internal)
            
            if not S:
                # If no internal edges available, we can only take 1 vertex
                # (A clique of size 2 {v, u} has no internal edge to cover, 
                #  but it's a valid clique. The "internal edge" constraint 
                #  only applies if |S| >= 2)
                # Actually, if |S|=1, there are no internal edges to cover.
                # So we can always take at least 1.
                u = next(iter(U_v))
                S = {u}
            
            # Form clique
            clique = [max_v] + list(S)
            cliques.append(clique)
            
            # Update uncovered sets
            # Remove cross edges for vertices in S
            for u in S:
                if max_v in uncovered_cross:
                    uncovered_cross[max_v].discard(u)
            
            # Remove internal edges within S
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u1 = S_list[i]
                    u2 = S_list[j]
                    e = (min(u1, u2), max(u1, u2))
                    uncovered_internal.discard(e)
            
            # Clean up empty cross_edges
            if not uncovered_cross[max_v]:
                del uncovered_cross[max_v]
                
        # Phase 2: Cover remaining internal edges
        # We need to partition the edges in uncovered_internal into cliques
        while uncovered_internal:
            # Pick an edge to start
            # Heuristic: Pick edge with high degree?
            # For now, just pick the first one
            e0 = next(iter(uncovered_internal))
            u, w = e0
            
            # Try to extend {u, w} to a maximal clique
            candidates = set()
            # Find common neighbors of u and w in uncovered_internal
            # Actually, we just need to find a clique containing u and w
            # Let's use the find_clique helper on the neighborhood?
            # Or just run find_clique on all vertices incident to remaining edges?
            
            # Vertices incident to any edge in uncovered_internal
            incident = set()
            for a, b in uncovered_internal:
                incident.add(a)
                incident.add(b)
                
            # We want a clique containing u and w.
            # Let's just find a maximal clique in the subgraph induced by incident
            # and check if it contains u and w? No, that might miss it.
            
            # Better:
            # Start with S = {u, w}
            S = {u, w}
            # Try to add other vertices from incident
            # Sort by degree in uncovered_internal
            deg = {v: 0 for v in incident}
            for a, b in uncovered_internal:
                deg[a] += 1
                deg[b] += 1
                
            sorted_incident = sorted(incident, key=lambda x: -deg[x])
            
            for x in sorted_incident:
                if x in S:
                    continue
                # Check if x can be added to S
                ok = True
                for y in S:
                    e = (min(x, y), max(x, y))
                    if e not in uncovered_internal:
                        ok = False
                        break
                if ok:
                    S.add(x)
            
            # Form clique
            cliques.append(list(S))
            
            # Remove all edges within S
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u1 = S_list[i]
                    u2 = S_list[j]
                    e = (min(u1, u2), max(u1, u2))
                    uncovered_internal.discard(e)
                    
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
