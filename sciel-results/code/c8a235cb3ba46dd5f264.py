import time
from collections import deque
import itertools
import sys

# Increase recursion limit for branch and bound
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85
    
    # 1. Pre-compute base graph properties
    base_adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    # Base distances
    base_dist = [[0]*n for _ in range(n)]
    for s in range(n):
        q = deque([s])
        base_dist[s][s] = 0
        while q:
            u = q.popleft()
            for v in base_adj[u]:
                if base_dist[s][v] == 0 and v != s:
                    base_dist[s][v] = base_dist[s][u] + 1
                    q.append(v)
    
    # 2. Identify candidate edges (Cross-parity to guarantee triangle-free)
    # Triangle-free condition: If we only add edges between even and odd vertices, 
    # and the base graph is bipartite (it is a path), the union remains bipartite and thus triangle-free.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_set:
                continue
            if u % 2 != v % 2:
                candidates.append((u, v))
    
    # Sort candidates by potential impact (longest distance first) to help pruning
    candidates.sort(key=lambda x: base_dist[x[0]][x[1]], reverse=True)
    m = len(candidates)
    
    # 3. Pre-compute distance improvements for each candidate edge
    # dist_imp[i][u][v] = max(0, base_dist[u][v] - dist_with_edge_i[u][v])
    # We only care about pairs where base_dist > 4
    # However, to be safe and accurate for the ILP, we compute full dist matrix for each edge? 
    # That's too slow. 
    # Optimization: Only consider pairs (u,v) with base_dist[u][v] > 4.
    # Let Critical_Pairs = [(u,v) where u<v and base_dist[u][v] > 4]
    
    critical_pairs = []
    for u in range(n):
        for v in range(u+1, n):
            if base_dist[u][v] > 4:
                critical_pairs.append((u, v))
                
    # If no critical pairs, 0 edges needed
    if not critical_pairs:
        return []
        
    # For each critical pair, determine which edges fix it.
    # An edge e=(a,b) fixes pair (u,v) if the shortest path using e is <= 4.
    # Since we are adding multiple edges, the "fix" condition is non-linear (OR of paths).
    # However, for a lower bound and initial pruning, we can use the property that 
    # if we add a set S of edges, dist(u,v) = min(base_dist[u][v], min_{e in S} dist_via_e(u,v)).
    # This is an approximation if paths can use multiple added edges. 
    # BUT, if we restrict to a small number of edges (which we expect), paths using multiple 
    # added edges are less likely to be the unique solution, or we can verify at the end.
    # Actually, for diameter 4, a path using 2 added edges would look like:
    # u --path-- a --edge-- b --path-- c --edge-- d --path-- v
    # Length = dist(u,a) + 1 + dist(b,c) + 1 + dist(d,v).
    # This is complex. Let's stick to the "single edge shortcut" model for the ILP constraint, 
    # but verify the full diameter at the end. If the ILP solution fails verification, 
    # we can fall back to a local search or increase the edge count.
    
    # For each critical pair p, and each candidate edge e, compute dist_via_e
    # dist_via_e(u, v) = min( dist(u, a) + 1 + dist(b, v), dist(u, b) + 1 + dist(a, v) )
    # Note: dist(u, a) here is the base distance. This is an under-estimate of the 
    # distance if other edges are present, but an over-estimate if we assume only this edge 
    # helps. Wait.
    # If we add edge (a,b), the new distance is min(base_dist, dist(u,a)+1+dist(b,v), ...).
    # If we add multiple edges, the distance can be even smaller.
    # So, if a set S satisfies "for all critical pairs, min_{e in S} dist_via_e <= 4", 
    # does it guarantee diameter <= 4?
    # Not necessarily, because a path might use TWO edges from S.
    # Example: u to a (1), a-b (1), b to c (1), c-d (1), d to v (1). Total 5.
    # If we have edges (a,b) and (c,d), and base dists are 1 each.
    # Base dist u to v might be 10.
    # Single edge shortcut via (a,b): 1+1+1+1+1 = 5 (using c-d as path? No, c-d is an edge).
    # If we treat c-d as base path, dist(b,v) includes c-d? No, base_dist doesn't include c-d.
    # So the "single edge shortcut" model underestimates the benefit of multiple edges.
    # It might force us to add MORE edges than necessary.
    # This is a conservative approach. It will find a valid solution, but maybe not minimal.
    # Given the small N, we can try to solve for k=1,2,3... using this conservative model.
    
    # Let's refine: 
    # For each critical pair (u,v), we need a set of edges S such that 
    # shortest_path(u,v, G+S) <= 4.
    
    # Heuristic ILP approach:
    # Variables x_e in {0,1}
    # Minimize sum(x_e)
    # Subject to: For each critical pair (u,v), there exists a path of length <= 4.
    # This is hard to linearize directly for multi-edge paths.
    
    # Alternative Strategy: Branch and Bound on the set of edges.
    # Since we expect a small number of edges (maybe 2-4), we can enumerate combinations.
    # But C(100, 4) is large.
    
    # Let's use the "Conservative Single-Edge Shortcut" model for the ILP constraints.
    # If the solution found is valid (verified by BFS), we return it.
    # If not, we relax the constraint or use a fallback search.
    
    # Pre-compute: for each critical pair p, which edges e can "fix" it individually?
    # fix[p][e] = 1 if dist_via_e(p) <= 4 else 0
    
    # To speed up, we only consider edges that are "useful".
    # An edge (a,b) is useful for (u,v) if min(dist(u,a)+1+dist(b,v), dist(u,b)+1+dist(a,v)) <= 4.
    
    # We will build a constraint matrix.
    # Rows: critical_pairs
    # Cols: candidates
    # Entry: 1 if edge fixes pair (via single shortcut), 0 otherwise.
    
    # Note: If an edge doesn't fix a pair via single shortcut, it might still help in a multi-edge path.
    # We will ignore this for the ILP, but verify later.
    
    constraint_matrix = []
    for u, v in critical_pairs:
        row = []
        for a, b in candidates:
            # Calculate shortest path using only this edge as shortcut
            # Path 1: u -> a -> b -> v
            # dist(u,a) + 1 + dist(b,v)
            # Path 2: u -> b -> a -> v
            # dist(u,b) + 1 + dist(a,v)
            
            d1 = base_dist[u][a] + 1 + base_dist[b][v]
            d2 = base_dist[u][b] + 1 + base_dist[a][v]
            
            if d1 <= 4 or d2 <= 4:
                row.append(1)
            else:
                row.append(0)
        constraint_matrix.append(row)
        
    # If a row is all zeros, no single edge can fix that pair.
    # This implies we MUST use a multi-edge path.
    # In that case, the ILP will fail to find a solution for k=1,2... using this model.
    # We need to detect this.
    
    has_unfixable_row = any(all(x == 0 for x in row) for row in constraint_matrix)
    
    if has_unfixable_row:
        # Fallback: Use a more robust search.
        # Since N=24 is small, we can do a BFS-based local search or random restarts 
        # with a stronger validation.
        # But first, let's try to solve the ILP for the fixable parts and see if 
        # the "unfixable" parts get fixed by the combination of edges.
        # Actually, if a pair (u,v) has distance > 4 in base, and no single edge 
        # reduces it to <= 4, it's possible that two edges do.
        # Example: u(0) to v(10). Base dist 10.
        # Edge (2,8). dist(0,2)=2, 2-8 (1), 8-10 (2). Total 5. Still > 4.
        # Edge (4,6). dist(0,4)=4, 4-6 (1), 6-10 (4). Total 9.
        # If we add BOTH (2,8) and (4,6).
        # Path 0->2->8->6->4... wait.
        # 0->2 (2), 2->8 (1), 8->6 (2), 6->4 (1), 4->...
        # This is getting complicated.
        
        # Given the time limit and complexity, let's switch to a 
        # **Greedy Set Cover with Verification** approach which is more robust 
        # to multi-edge paths, but guided by the ILP-like scoring.
        
        # Score an edge by how many critical pairs it "reduces" significantly.
        # Or better: Just use the existing greedy but with a better heuristic:
        # Pick the edge that minimizes the maximum remaining distance.
        
        best_solution = None
        best_count = float('inf')
        
        def _build_adj(added):
            adj = {i: set(base_adj[i]) for i in range(n)}
            for u, v in added:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        def _fast_diameter(adj, limit=4):
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            if dist[v] > limit:
                                return dist[v]
                            q.append(v)
                if any(d == -1 for d in dist):
                    return 999
            return 0
            
        def greedy_search(cand_list, max_edges=8):
            added = []
            adj = {i: set(base_adj[i]) for i in range(n)}
            cur_diam = _fast_diameter(adj, 4)
            if cur_diam <= 4:
                return []
            
            remaining = list(cand_list)
            
            for _ in range(max_edges):
                if time.time() > deadline - 2:
                    break
                best_edge = None
                best_new_diam = cur_diam
                
                for u, v in remaining:
                    if v in adj[u]:
                        continue
                    # Triangle-free check
                    if adj[u] & adj[v]:
                        continue
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    new_diam = _fast_diameter(adj, 4)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    if new_diam < best_new_diam:
                        best_new_diam = new_diam
                        best_edge = (u, v)
                    
                    if new_diam <= 4:
                        break
                
                if best_edge is None:
                    break
                
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                added.append((u, v))
                cur_diam = best_new_diam
                remaining = [e for e in remaining if e != (u, v)]
                
                if cur_diam <= 4:
                    break
            
            if cur_diam <= 4:
                return added
            return None

        # Run greedy with sorted candidates
        sol = greedy_search(candidates)
        if sol:
            best_solution = sol
            best_count = len(sol)
            
        # Randomized restarts
        import random
        for _ in range(20):
            if time.time() > deadline - 2: break
            shuffled = candidates[:]
            random.shuffle(shuffled)
            sol = greedy_search(shuffled)
            if sol and len(sol) < best_count:
                best_count = len(sol)
                best_solution = sol
                if best_count <= 2: break
                
        if best_solution:
            return [[int(u), int(v)] for u, v in best_solution]
        else:
            return [] # Should not happen

    # If all rows are fixable by some single edge, we can use ILP/Backtracking
    # to find the minimum set cover.
    
    # Backtracking search for Set Cover
    # Sort columns (edges) by number of 1s (descending) to prune better
    # Actually, for set cover, picking edges that cover many uncovered sets is good.
    
    # State: (uncovered_pairs_set, current_edges, depth)
    # We want to find min depth.
    
    # Pre-compute for each edge, which pairs it covers
    edge_covers = [set() for _ in range(m)]
    for i, row in enumerate(constraint_matrix):
        for j, val in enumerate(row):
            if val == 1:
                edge_covers[j].add(i)
                
    # Sort edges by coverage size
    edge_indices = sorted(range(m), key=lambda x: len(edge_covers[x]), reverse=True)
    
    best_k = float('inf')
    best_edges = None
    
    def backtrack(uncovered, current_edges, depth):
        nonlocal best_k, best_edges
        
        if time.time() > deadline - 2:
            return
            
        if depth >= best_k:
            return
            
        if not uncovered:
            if depth < best_k:
                best_k = depth
                best_edges = current_edges[:]
            return
            
        # Heuristic: Pick the uncovered pair with the fewest remaining edges that can cover it
        # This is a standard CSP heuristic (Minimum Remaining Values)
        
        # Find the pair in 'uncovered' that has the fewest candidates in 'edge_indices' 
        # that are not yet chosen and cover it.
        
        # To speed up, we can just pick the first uncovered pair and try its covering edges.
        # Or better, pick the pair with minimum options.
        
        min_options = float('inf')
        target_pair = None
        target_edges = []
        
        for p in uncovered:
            # Find edges that cover p
            covering_edges = [e for e in edge_indices if p in edge_covers[e]]
            if len(covering_edges) < min_options:
                min_options = len(covering_edges)
                target_pair = p
                target_edges = covering_edges
                if min_options == 0:
                    return # Dead end
                
        # Try each edge that covers target_pair
        for e in target_edges:
            if e in current_edges:
                continue
                
            new_uncovered = uncovered - edge_covers[e]
            new_edges = current_edges + [e]
            backtrack(new_uncovered, new_edges, depth + 1)
            
    initial_uncovered = set(range(len(critical_pairs)))
    backtrack(initial_uncovered, [], 0)
    
    if best_edges:
        # Verify the solution
        sol = [candidates[e] for e in best_edges]
        adj = {i: set(base_adj[i]) for i in range(n)}
        for u, v in sol:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        valid = True
        for u in range(n):
            nbrs = list(adj[u])
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    if nbrs[j] in adj[nbrs[i]]:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            # Check diameter
            diam = _fast_diameter(adj, 4)
            if diam <= 4:
                return [[int(u), int(v)] for u, v in sol]
    
    # Fallback if ILP/Backtracking failed or verification failed
    # Use the greedy result from the other branch or a simple fallback
    # Since we didn't run greedy in this branch, we should have run it as a baseline.
    
    # Let's re-run greedy just in case
    def _fast_diameter(adj, limit=4):
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > limit:
                            return dist[v]
                        q.append(v)
            if any(d == -1 for d in dist):
                return 999
        return 0
        
    def greedy_search(cand_list, max_edges=8):
        added = []
        adj = {i: set(base_adj[i]) for i in range(n)}
        cur_diam = _fast_diameter(adj, 4)
        if cur_diam <= 4:
            return []
        
        remaining = list(cand_list)
        
        for _ in range(max_edges):
            if time.time() > deadline - 2:
                break
            best_edge = None
            best_new_diam = cur_diam
            
            for u, v in remaining:
                if v in adj[u]:
                    continue
                if adj[u] & adj[v]:
                    continue
                
                adj[u].add(v)
                adj[v].add(u)
                new_diam = _fast_diameter(adj, 4)
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                
                if new_diam <= 4:
                    break
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            added.append((u, v))
            cur_diam = best_new_diam
            remaining = [e for e in remaining if e != (u, v)]
            
            if cur_diam <= 4:
                break
        
        if cur_diam <= 4:
            return added
        return None

    sol = greedy_search(candidates)
    if sol:
        return [[int(u), int(v)] for u, v in sol]
        
    # Final fallback
    return []

# EVOLVE-BLOCK-END
