"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [list() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Precompute all paths of length 3 between all pairs
    # path_3[u][v] is a list of vertices w such that u-w-v is a path
    path_3 = [[[] for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v: continue
            if v in adj[u]:
                # Direct edge, no length 3 path needed, but we can store empty
                continue
            # Find w such that u-w and w-v
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    path_3[u][v].append(w)
            # Note: u-w-v is simple if w != u, w != v.
            
    # Precompute all paths of length 4 between all pairs?
    # Might be too many. Let's stick to length 3 primarily, and maybe 4 if needed.
    # Given the tight budget, length 3 is crucial.
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Candidate vertices: top degree
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    best_result = None
    
    # Try k from 9 down to 2
    for k in range(9, 1, -1):
        if time.time() > deadline:
            break
            
        # Generate combinations from top 15 vertices
        # If k is large, restrict candidate pool to avoid explosion
        pool_size = min(15, n)
        candidates = sorted_vertices[:pool_size]
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
                
            B = list(combo)
            B_set = set(B)
            
            # Check if subdivision is possible
            # 1. Identify non-adjacent pairs
            non_adj_pairs = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v not in adj[u]:
                        non_adj_pairs.append((u, v))
                        
            # Pigeonhole check
            available_vertices = n - k
            if len(non_adj_pairs) > available_vertices:
                continue
                
            # 2. CSP Solver
            # We need to assign a distinct internal vertex (or set) to each non-adj pair.
            # Since we prefer length 3, we assign a single vertex w.
            # If no length 3 path is available, we might need length 4 (2 vertices).
            # Let's try to assign length 3 paths first.
            # If that fails, we can relax to length 4, but it's more complex.
            # Given the budget, let's assume length 3 is sufficient for the optimal solution.
            # If not, we might miss a solution, but it's a good heuristic.
            
            # Actually, we can just check if there is a matching in the bipartite graph
            # where left nodes are non_adj_pairs and right nodes are available vertices,
            # and edge exists if w is a valid internal vertex for the pair (u,v).
            # This is a bipartite matching problem!
            # If we find a perfect matching, we have our length-3 paths.
            # If we don't, we might still have a solution with longer paths, but
            # given the constraints, this is the most likely case for high k.
            
            # Let's use Hopcroft-Karp or just simple DFS matching since sizes are small.
            # Left: non_adj_pairs
            # Right: V \ B
            
            # Build adjacency for matching
            # pair_idx -> list of available vertices
            pair_to_verts = {}
            for idx, (u, v) in enumerate(non_adj_pairs):
                # Valid w must be in V \ B and w in adj[u] and w in adj[v]
                valid_ws = [w for w in path_3[u][v] if w not in B_set]
                pair_to_verts[idx] = valid_ws
                
            # If any pair has no valid w, matching is impossible (for length 3)
            possible = True
            for idx in range(len(non_adj_pairs)):
                if not pair_to_verts[idx]:
                    possible = False
                    break
            
            if not possible:
                continue
                
            # Perform Bipartite Matching
            # We want to assign a unique vertex to each pair.
            match = [-1] * len(non_adj_pairs) # match[i] = vertex assigned to pair i
            
            def dfs_matching(v, seen):
                for u in pair_to_verts[v]:
                    if seen[u]:
                        continue
                    seen[u] = True
                    if match[u] == -1 or dfs_matching(match[u], seen):
                        match[u] = v
                        return True
                return False
            
            # Standard Kuhn's algorithm
            # We need to map vertices to indices? No, just use vertex id.
            # But match is indexed by pair index.
            # Let's restructure:
            # match_vertex[v] = pair_idx assigned to vertex v
            match_vertex = {v: -1 for v in range(n)}
            
            def try_kuhn(v, seen):
                # v is a pair index
                for u in pair_to_verts[v]:
                    if seen[u]:
                        continue
                    seen[u] = True
                    if match_vertex[u] == -1 or try_kuhn(match_vertex[u], seen):
                        match_vertex[u] = v
                        return True
                return False
            
            found_matching = True
            for i in range(len(non_adj_pairs)):
                seen = {v: False for v in range(n)}
                if not try_kuhn(i, seen):
                    found_matching = False
                    break
                    
            if not found_matching:
                continue
                
            # If matching found, we have a valid subdivision using length-3 paths for non-adj pairs
            # and direct edges for adj pairs.
            
            # Construct the result
            paths = []
            # Direct edges
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v in adj[u]:
                        paths.append([u, v])
                        
            # Length 3 paths from matching
            # We need to map pair (u,v) back to its index in non_adj_pairs
            # And get the matched vertex.
            # This is slightly tricky because match_vertex maps vertex -> pair_idx.
            # We need pair_idx -> vertex.
            pair_to_match_vert = {}
            for v, p_idx in match_vertex.items():
                if p_idx != -1:
                    pair_to_match_vert[p_idx] = v
            
            for idx, (u, v) in enumerate(non_adj_pairs):
                w = pair_to_match_vert[idx]
                paths.append([u, w, v])
                
            # Verify validity (just in case)
            # 1. Branches distinct? Yes, from combination.
            # 2. Path endpoints correct? Yes.
            # 3. Interiors disjoint?
            #    - Direct edges have empty interior.
            #    - Length 3 paths have interior {w}.
            #    - Matching ensures all w are distinct.
            #    - w not in B (checked in valid_ws).
            #    - So interiors are disjoint and avoid B.
            # 4. Consecutive vertices adjacent?
            #    - u-w: w in adj[u] (from path_3).
            #    - w-v: w in adj[v] (from path_3).
            
            result = {'branches': B, 'paths': paths}
            return result
            
    # Fallback to previous logic if k=9..2 fails?
    # The previous logic found k=7.
    # Let's try to find k=7 with a more relaxed path length if needed?
    # Or just return the best clique if no subdivision found.
    
    # If we didn't find anything, fall back to max clique
    # (Re-using Bron-Kerbosch from seed)
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    
    if len(best_clique) >= 2:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': list(branches), 'paths': paths}
        
    # Last resort
    u, v = P['edges'][0]
    return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
