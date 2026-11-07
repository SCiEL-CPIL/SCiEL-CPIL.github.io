"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute common neighbors for length-3 paths
    common = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            cn = adj[u] & adj[v]
            cn.discard(u)
            cn.discard(v)
            common[u][v] = cn

    # Precompute shortest paths (BFS) between all pairs, avoiding a given set
    # shortest_path[u][v] = list of shortest paths (as tuples of vertices including endpoints)
    # We'll compute on demand to save memory.
    
    def bfs_shortest_paths(s, t, forbidden):
        """Find all shortest simple paths from s to t avoiding forbidden vertices (except s,t)."""
        if s == t:
            return []
        if t in forbidden and t != s:
            return []
        # BFS to find distance
        dist = {s: 0}
        parent = {s: []}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for v in adj[u]:
                if v in forbidden and v != t:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = [u]
                    q.append(v)
                elif dist[v] == dist[u] + 1:
                    parent[v].append(u)
        if t not in dist:
            return []
        # Backtrack to enumerate all shortest paths
        paths = []
        def backtrack(node, path):
            if node == s:
                paths.append(tuple(reversed(path)))
                return
            for p in parent[node]:
                if p in forbidden and p != s:
                    continue
                backtrack(p, path + [p])
        backtrack(t, [t])
        return paths
    
    # Bron-Kerbosch for max clique (baseline)
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
    
    # If max clique is large enough, use it directly
    if len(best_clique) >= 7:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}
    
    # Core solver: given a candidate branch set B, try to find disjoint paths
    # using a combination of length-2 (direct), length-3 (one internal), and BFS paths
    def try_subdivision(B):
        """Try to find a valid subdivision for branch set B.
        Returns paths list or None."""
        k = len(B)
        B_set = set(B)
        
        # Identify all pairs and their adjacency status
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append((i, j, B[i] in adj[B[j]]))
        
        non_adj_pairs = [(i, j) for i, j, is_adj in pairs if not is_adj]
        
        # Pigeonhole: need at least as many available vertices as non-adj pairs
        if len(non_adj_pairs) > n - k:
            return None
        
        # Strategy 1: Try length-3 paths via bipartite matching (fast)
        # Each non-adj pair gets one unique internal vertex
        pair_to_verts = {}
        for idx, (i, j) in enumerate(non_adj_pairs):
            valid = [w for w in common[B[i]][B[j]] if w not in B_set]
            pair_to_verts[idx] = valid
            if not valid:
                return None
        
        match_vertex = {}
        
        def try_kuhn(pair_idx, seen):
            for w in pair_to_verts[pair_idx]:
                if w in seen:
                    continue
                seen.add(w)
                if w not in match_vertex or try_kuhn(match_vertex[w], seen):
                    match_vertex[w] = pair_idx
                    return True
            return False
        
        matching_found = True
        for idx in range(len(non_adj_pairs)):
            if not try_kuhn(idx, set()):
                matching_found = False
                break
        
        if matching_found:
            # Build lookup
            pair_internal = {}
            for w, pair_idx in match_vertex.items():
                i, j = non_adj_pairs[pair_idx]
                pair_internal[(i, j)] = w
            
            paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    if B[j] in adj[B[i]]:
                        paths.append([B[i], B[j]])
                    else:
                        w = pair_internal[(i, j)]
                        paths.append([B[i], w, B[j]])
            return paths
        
        # Strategy 2: BFS-based disjoint path routing with backtracking
        # For each non-adj pair, find shortest paths avoiding B and already-used vertices
        # Order pairs by fewest options first (MRV heuristic)
        def bfs_route(B, non_adj_pairs):
            # Sort pairs by number of available common neighbors (fewest first)
            pair_options = []
            for idx, (i, j) in enumerate(non_adj_pairs):
                valid = [w for w in common[B[i]][B[j]] if w not in B_set]
                pair_options.append((len(valid), idx, i, j, valid))
            pair_options.sort()
            
            # Backtracking search
            used_internal = set()
            assignment = {}
            
            def backtrack(pos):
                if pos == len(pair_options):
                    return True
                _, idx, i, j, valid = pair_options[pos]
                # Try each candidate internal vertex
                for w in valid:
                    if w in used_internal:
                        continue
                    # Check if w is still valid (not used)
                    used_internal.add(w)
                    assignment[(i, j)] = w
                    if backtrack(pos + 1):
                        return True
                    used_internal.remove(w)
                    del assignment[(i, j)]
                return False
            
            if backtrack(0):
                paths = []
                for i in range(k):
                    for j in range(i + 1, k):
                        if B[j] in adj[B[i]]:
                            paths.append([B[i], B[j]])
                        else:
                            w = assignment[(i, j)]
                            paths.append([B[i], w, B[j]])
                return paths
            return None
        
        result = bfs_route(B, non_adj_pairs)
        return result
    
    # Search for the largest valid branch set
    best_result = None
    
    for k in range(12, 2, -1):
        if time.time() > deadline:
            break
        
        # Candidate pool: expand for smaller k to find valid sets
        if k >= 10:
            pool_size = min(n, k + 4)
        elif k >= 8:
            pool_size = min(n, k + 6)
        else:
            pool_size = min(n, k + 10)
        candidates = sorted_vertices[:pool_size]
        
        # Generate combinations with early pruning
        found = False
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            
            B = list(combo)
            B_set = set(B)
            non_adj_count = 0
            for i in range(k):
                for j in range(i + 1, k):
                    if B[j] not in adj[B[i]]:
                        non_adj_count += 1
            if non_adj_count > n - k:
                continue
            
            result = try_subdivision(B)
            if result is not None:
                best_result = (B, result)
                found = True
                break
        
        if found:
            break
        
        # If we didn't find with the current pool, try a wider pool for this k
        if k <= 9 and pool_size < n:
            wider_pool = sorted_vertices[:min(n, pool_size + 4)]
            for combo in combinations(wider_pool, k):
                if time.time() > deadline:
                    break
                B = list(combo)
                B_set = set(B)
                non_adj_count = 0
                for i in range(k):
                    for j in range(i + 1, k):
                        if B[j] not in adj[B[i]]:
                            non_adj_count += 1
                if non_adj_count > n - k:
                    continue
                result = try_subdivision(B)
                if result is not None:
                    best_result = (B, result)
                    found = True
                    break
            if found:
                break
    
    if best_result is not None:
        branches, paths = best_result
        branches = [int(b) for b in branches]
        paths = [[int(v) for v in p] for p in paths]
        return {'branches': branches, 'paths': paths}
    
    # Fallback: use max clique
    if len(best_clique) >= 3:
        branches = sorted([int(b) for b in best_clique])
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': branches, 'paths': paths}
    
    # Fallback: triangle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    for a, b, c in combinations(range(n), 3):
        if (min(a, b), max(a, b)) in edge_set and \
           (min(a, c), max(a, c)) in edge_set and \
           (min(b, c), max(b, c)) in edge_set:
            branches = sorted([a, b, c])
            paths = [[branches[0], branches[1]], [branches[0], branches[2]], [branches[1], branches[2]]]
            return {'branches': branches, 'paths': paths}
    
    # Last resort
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
