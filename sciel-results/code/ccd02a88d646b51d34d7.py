"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # 1. Setup Base Graph
    base_adj = [[] for _ in range(n)]
    base_edge_set = set()
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        base_edge_set.add((min(u, v), max(u, v)))

    # 2. Identify Candidate Edges
    # Candidates are pairs (u, v) not in base, and not forming a triangle with base edges.
    # A triangle with base edges occurs if dist_base(u, v) == 2.
    # Since base is a path, dist_base(u, v) == abs(u - v).
    # So candidates are pairs with abs(u - v) > 2.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            if abs(u - v) == 2:
                continue # Forms triangle with base edge (u, u+1) and (u+1, v) or similar
            candidates.append((u, v))
            
    # Sort candidates by "usefulness" heuristic: 
    # Longer jumps are generally more useful for diameter.
    # We sort by distance descending to try "big" edges first.
    candidates.sort(key=lambda e: abs(e[0] - e[1]), reverse=True)
    
    # 3. Helper Functions
    
    def get_diameter(adj):
        max_d = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
                        if dist[nei] > max_d:
                            max_d = dist[nei]
        return max_d

    def is_triangle_free(adj, added_edges):
        # Check if the current graph (base + added) is triangle-free.
        # We only need to check triangles involving at least one added edge.
        # A triangle (u, v, w) exists if:
        # 1. All 3 are base: Impossible.
        # 2. 2 base, 1 added: We filtered candidates so this is impossible.
        #    (Candidate (u,v) with dist 2 is excluded. If dist > 2, no base edge connects u and v's neighbors to form triangle? 
        #     Wait. Triangle (u, v, w). If (u,v) is added.
        #     If (u,w) and (v,w) are base, then w is neighbor of u and v in base.
        #     In a path, u and v share a neighbor w only if |u-v|=2.
        #     So yes, filtering dist=2 handles 2-base-1-added triangles.)
        # 3. 1 base, 2 added: 
        #    Triangle (u, v, w) where (u,v) is base.
        #    Then (u,w) and (v,w) must be added.
        #    So we check if any added edge (u,w) and (v,w) exist for any base edge (u,v).
        # 4. 0 base, 3 added:
        #    Triangle (u, v, w) where all edges are added.
        #    Check if any 3 added edges form a triangle.
        
        # Build adjacency set for fast lookup
        adj_set = set()
        for u in range(n):
            for v in adj[u]:
                adj_set.add((min(u, v), max(u, v)))
        
        # Check 1-base-2-added
        for u, v in base_edges:
            # Check all w
            for w in range(n):
                if w == u or w == v:
                    continue
                if (min(u, w), max(u, w)) in adj_set and (min(v, w), max(v, w)) in adj_set:
                    # Check if these are added edges
                    # If they are base, we already handled 2-base case (impossible for dist>2 candidates? No, w could be such that (u,w) is base)
                    # But if (u,w) is base and (v,w) is base, then u,v,w form triangle in base? No, base is path.
                    # So if (u,w) and (v,w) are in adj_set, and (u,v) is base:
                    # If (u,w) is added and (v,w) is added -> Violation.
                    # If one is base and one is added -> Violation? 
                    # If (u,w) is base, then w is neighbor of u. In path, neighbors of u are u-1, u+1.
                    # So w = u-1 or u+1.
                    # If w = u+1, then v must be u-1 or u+2?
                    # If (u,v) is base, v = u-1 or u+1.
                    # If v = u+1, w = u+1 -> w=v, invalid.
                    # If v = u-1, w = u+1. Then (v,w) = (u-1, u+1). dist=2. This is a candidate? No, dist=2 is excluded.
                    # So (v,w) cannot be a candidate if it's dist 2.
                    # So if (u,w) is base, (v,w) cannot be a candidate.
                    # Thus, the only way to have 1-base-2-added triangle is if both (u,w) and (v,w) are added.
                    
                    if (min(u, w), max(u, w)) not in base_edge_set and (min(v, w), max(v, w)) not in base_edge_set:
                        return False
        
        # Check 3-added
        # For each added edge (u, v), check if they share a neighbor w such that (u,w) and (v,w) are also added.
        added_set = set((min(u, v), max(u, v)) for u, v in added_edges)
        for u, v in added_edges:
            # Find common neighbors in the FULL graph
            common = set(adj[u]) & set(adj[v])
            for w in common:
                if (min(u, w), max(u, w)) in added_set and (min(v, w), max(v, w)) in added_set:
                    return False
                    
        return True

    def would_create_triangle(new_edge, current_added):
        u, v = new_edge
        # Check against current_added
        # 1. Does (u,v) form a triangle with two edges in current_added?
        #    i.e., exists w such that (u,w) and (v,w) are in current_added.
        added_set = set((min(a, b), max(a, b)) for a, b in current_added)
        common = set(base_adj[u]) | set(a for a, b in current_added if a == u or b == u)
        common_v = set(base_adj[v]) | set(a for a, b in current_added if a == v or b == v)
        # Actually, just check all w
        for w in range(n):
            if w == u or w == v:
                continue
            e1 = (min(u, w), max(u, w))
            e2 = (min(v, w), max(v, w))
            
            # Case 1: Both e1, e2 are added
            if e1 in added_set and e2 in added_set:
                return True
            
            # Case 2: One added, one base
            # If e1 is added and e2 is base (or vice versa), does it form a triangle?
            # Triangle (u, v, w). Edges: (u,v) [new], (u,w), (v,w).
            # If (u,w) is added and (v,w) is base:
            #   Then (v,w) is a base edge. w is neighbor of v in path.
            #   So w = v-1 or v+1.
            #   For (u,w) to be a valid candidate, dist(u,w) > 2.
            #   Does this form a triangle? Yes, geometrically.
            #   Is it forbidden? Yes, triangle-free graph.
            #   So if (u,w) is in current_added and (v,w) is in base_edge_set, return True.
            if e1 in added_set and e2 in base_edge_set:
                return True
            if e2 in added_set and e1 in base_edge_set:
                return True
                
            # Case 3: Both base
            # If e1, e2 are base, then (u,v) would form a triangle with base.
            # We already filtered candidates with dist=2, so this shouldn't happen for valid candidates.
            # But just in case:
            if e1 in base_edge_set and e2 in base_edge_set:
                return True

        return False

    # 4. Branch and Bound
    
    best_solution = None
    best_count = float('inf')
    
    # Heuristic to get initial upper bound
    # Greedy: Add edge that reduces diameter most, if no triangle.
    def greedy_initial():
        adj = [list(neigh) for neigh in base_adj]
        added = []
        edge_set = set(base_edge_set)
        
        while True:
            d = get_diameter(adj)
            if d <= 4:
                break
            best_e = None
            best_score = -1
            # Precompute dists for scoring
            dists = []
            for i in range(n):
                dist = [-1] * n
                dist[i] = 0
                q = deque([i])
                while q:
                    node = q.popleft()
                    for nei in adj[node]:
                        if dist[nei] == -1:
                            dist[nei] = dist[node] + 1
                            q.append(nei)
                dists.append(dist)
            
            for e in candidates:
                u, v = e
                if (u, v) in edge_set:
                    continue
                if would_create_triangle(e, added):
                    continue
                # Score: reduction in bad pairs
                new_bad = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if dists[i][j] > 4:
                            new_d = min(dists[i][j], 
                                        dists[i][u] + 1 + dists[v][j],
                                        dists[i][v] + 1 + dists[u][j])
                            if new_d > 4:
                                new_bad += 1
                # Heuristic: prefer edges that are "central" or reduce max dist
                # Just use -new_bad as score (minimize bad)
                score = -new_bad
                if score > best_score:
                    best_score = score
                    best_e = e
                    
            if best_e is None:
                break
            u, v = best_e
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            added.append([u, v])
            
        return added

    init_sol = greedy_initial()
    if init_sol:
        best_count = len(init_sol)
        best_solution = [list(e) for e in init_sol]
    else:
        # Fallback: just return empty if greedy fails (shouldn't happen)
        return []

    # B&B Search
    # We search for a solution with size < best_count
    # State: idx (index in candidates), current_added (list)
    
    def dfs(idx, current_added):
        nonlocal best_count, best_solution
        
        if len(current_added) >= best_count:
            return
            
        # Prune: If we can't reach diameter 4 with remaining edges? 
        # Hard to estimate.
        
        # Check if current state is valid (diameter <= 4)
        # Only check if we've considered enough edges? 
        # Actually, we can check diameter at any node.
        # If diameter <= 4, we have a solution.
        adj = [list(neigh) for neigh in base_adj]
        for u, v in current_added:
            adj[u].append(v)
            adj[v].append(u)
            
        d = get_diameter(adj)
        if d <= 4:
            # Valid solution
            if len(current_added) < best_count:
                best_count = len(current_added)
                best_solution = [list(e) for e in current_added]
            return # Don't add more edges

        if idx == len(candidates):
            return

        # Option 1: Skip candidate[idx]
        dfs(idx + 1, current_added)
        
        # Option 2: Add candidate[idx]
        e = candidates[idx]
        if not would_create_triangle(e, current_added):
            new_added = current_added + [list(e)]
            dfs(idx + 1, new_added)

    # Start search
    dfs(0, [])
    
    if best_solution:
        return best_solution
    return init_sol

# EVOLVE-BLOCK-END
