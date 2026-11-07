import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency structures
        # K is 0..k-1, I is k..n-1
        adj_K = [set() for _ in range(k)]
        adj_I_to_K = [set() for _ in range(n)] # Only valid for i >= k
        
        cross_edges = []
        K_edges = []
        
        for u, v in edges:
            if u < k and v < k:
                adj_K[u].add(v)
                adj_K[v].add(u)
                K_edges.append((u, v))
            elif u >= k and v < k:
                adj_I_to_K[u].add(v)
                cross_edges.append((u, v))
            elif v >= k and u < k:
                adj_I_to_K[v].add(u)
                cross_edges.append((v, u))

        # State: uncovered edges
        uncovered_cross = {i: set() for i in range(k, n)}
        for u, v in cross_edges:
            uncovered_cross[u].add(v)
            
        uncovered_K = set(K_edges)
        
        cliques = []
        
        # Helper: check if S is a clique in K
        def is_clique_in_K(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    if S_list[j] not in adj_K[S_list[i]]:
                        return False
            return True

        # Helper: get gain for a clique S containing v (v in I)
        def gain_with_v(v, S):
            # Cross edges covered
            g = 0
            for u in S:
                if u in uncovered_cross[v]:
                    g += 1
            # K edges covered
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    e = (min(u, w), max(u, w))
                    if e in uncovered_K:
                        g += 1
            return g

        # Helper: get gain for a clique S in K only
        def gain_K_only(S):
            g = 0
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    e = (min(u, w), max(u, w))
                    if e in uncovered_K:
                        g += 1
            return g

        # Greedy clique construction for a given v in I
        def find_best_S_for_v(v):
            # Candidates are neighbors of v in K that have uncovered cross edge to v
            # Actually, even if cross edge is covered, we might still want u to cover K edges?
            # No, if (v,u) is covered, adding u to S only helps if it covers K edges.
            # But if (v,u) is covered, the clique {v} U S is still valid.
            # However, if we don't use u to cover the cross edge, maybe we can use u in a K-only clique?
            # The global greedy compares options.
            # For this heuristic, we consider all neighbors u in K.
            
            candidates = list(adj_I_to_K[v])
            if not candidates:
                return set(), 0
            
            S = set()
            best_gain = 0
            
            # Greedy expansion
            while True:
                best_u = None
                best_marginal = 0
                for u in candidates:
                    if u in S:
                        continue
                    # Check if S U {u} is a clique
                    # u must be adjacent to all in S
                    valid = True
                    for w in S:
                        if u not in adj_K[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate marginal gain
                    # Cross edge (v, u)
                    mg = 0
                    if u in uncovered_cross[v]:
                        mg += 1
                    
                    # K edges (u, w) for w in S
                    for w in S:
                        e = (min(u, w), max(u, w))
                        if e in uncovered_K:
                            mg += 1
                    
                    if mg > best_marginal:
                        best_marginal = mg
                        best_u = u
                
                if best_u is None:
                    break
                
                S.add(best_u)
                best_gain += best_marginal
                
                if time.time() > deadline - 5.0:
                    break

            return S, best_gain

        # Greedy clique construction for K only
        def find_best_S_K_only():
            # We want to find a clique S in K maximizing uncovered K edges.
            # Heuristic: Start with empty, add vertex with max degree in uncovered K subgraph?
            # Or pick an edge, expand?
            
            if not uncovered_K:
                return set(), 0
            
            # Get active vertices in K
            active_K = set()
            for u, v in uncovered_K:
                active_K.add(u)
                active_K.add(v)
                
            if not active_K:
                return set(), 0

            S = set()
            best_gain = 0
            
            # Try starting with each edge? Or just greedy on vertices?
            # Greedy on vertices:
            # 1. Pick vertex u with max degree in uncovered K graph.
            # 2. Add u to S.
            # 3. Restrict to neighbors of u in uncovered K graph.
            # 4. Repeat.
            
            # To avoid bias, we can try a few random starts or just the global max degree.
            # Let's do a simple greedy:
            candidates = list(active_K)
            
            while candidates:
                # Pick u with max degree in current candidate set (considering only uncovered edges)
                # Degree in uncovered subgraph induced by candidates
                best_u = None
                max_deg = -1
                for u in candidates:
                    deg = 0
                    for v in candidates:
                        if u != v:
                            e = (min(u, v), max(u, v))
                            if e in uncovered_K:
                                deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        best_u = u
                
                if best_u is None or max_deg == 0:
                    break
                
                S.add(best_u)
                candidates = [v for v in candidates if v != best_u and best_u in adj_K[v]]
                
                if time.time() > deadline - 5.0:
                    break
            
            if not S:
                return set(), 0
                
            return S, gain_K_only(S)

        # Main Loop
        while uncovered_cross or uncovered_K:
            if time.time() > deadline - 10.0:
                # Fallback: cover remaining edges with pairs
                for v in range(k, n):
                    for u in list(uncovered_cross[v]):
                        cliques.append([v, u])
                        uncovered_cross[v].remove(u)
                for u, v in list(uncovered_K):
                    cliques.append([u, v])
                    uncovered_K.remove((u, v))
                break

            best_clique = None
            best_gain = 0
            best_type = None # 'I' or 'K'
            best_v = None
            
            # Check I-cliques
            for v in range(k, n):
                if not uncovered_cross[v]:
                    continue
                S, g = find_best_S_for_v(v)
                if g > best_gain:
                    best_gain = g
                    best_clique = S
                    best_type = 'I'
                    best_v = v
            
            # Check K-clique
            S_k, g_k = find_best_S_K_only()
            if g_k > best_gain:
                best_gain = g_k
                best_clique = S_k
                best_type = 'K'
                best_v = None
            
            if best_gain == 0:
                # Should not happen, but safety
                # Pick any uncovered edge
                if uncovered_cross:
                    v = next(iter([v for v in range(k, n) if uncovered_cross[v]]))
                    u = next(iter(uncovered_cross[v]))
                    best_clique = {u}
                    best_type = 'I'
                    best_v = v
                    best_gain = 1
                elif uncovered_K:
                    u, v = next(iter(uncovered_K))
                    best_clique = {u, v}
                    best_type = 'K'
                    best_v = None
                    best_gain = 1
            
            # Add clique
            if best_type == 'I':
                clique_list = [best_v] + list(best_clique)
                cliques.append(clique_list)
                
                # Update uncovered
                for u in best_clique:
                    if u in uncovered_cross[best_v]:
                        uncovered_cross[best_v].remove(u)
                
                # Remove K edges within best_clique
                S_list = list(best_clique)
                edges_to_remove = set()
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        u, w = S_list[i], S_list[j]
                        e = (min(u, w), max(u, w))
                        if e in uncovered_K:
                            edges_to_remove.add(e)
                uncovered_K -= edges_to_remove
            else: # K
                clique_list = list(best_clique)
                cliques.append(clique_list)
                
                # Remove K edges within best_clique
                S_list = list(best_clique)
                edges_to_remove = set()
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        u, w = S_list[i], S_list[j]
                        e = (min(u, w), max(u, w))
                        if e in uncovered_K:
                            edges_to_remove.add(e)
                uncovered_K -= edges_to_remove

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        res = solve_one(*spec)
        # Ensure types are correct
        clean_res = []
        for clique in res:
            # Convert to list of ints
            clean_clique = [int(x) for x in clique]
            if len(clean_clique) >= 2:
                clean_res.append(clean_clique)
        results.append(clean_res)

    return results
