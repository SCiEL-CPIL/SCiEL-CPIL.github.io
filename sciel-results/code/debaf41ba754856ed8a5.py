import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    start_time = time.time()
    deadline = start_time + 85
    
    # All added edges must be bipartite (even-odd) to stay triangle-free
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    bipartite_candidates = []
    for u in even_verts:
        for v in odd_verts:
            key = (min(u,v), max(u,v))
            if key not in base_edges:
                bipartite_candidates.append(key)
    
    m = len(bipartite_candidates)
    
    # For each candidate edge, compute which far pairs it "covers"
    # A far pair (i,j) with |i-j|>4 is covered by edge (u,v) if
    # the shortest path using that edge gives dist <= 4
    # dist via edge = min(
    #   min(|i-u|, |i-v|) + 1 + min(|j-u|, |j-v|),
    #   min(|i-v|, |i-u|) + 1 + min(|j-v|, |j-u|)
    # )
    # But we also need the full graph to have diameter <= 4, which is more complex.
    # However, for a necessary condition: each far pair must be "covered" by at least one edge.
    
    # Far pairs
    far_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if j - i > 4:
                far_pairs.append((i, j))
    
    # For each candidate edge, determine which far pairs it can help
    # Edge (u,v) helps pair (i,j) if:
    # min(|i-u|+1+|j-v|, |i-v|+1+|j-u|) <= 4
    # AND the path from i to u (or v) and from v (or u) to j doesn't create issues
    # Actually in a path, dist(i,u) = |i-u|, so the shortcut distance is:
    # min(|i-u| + 1 + |j-v|, |i-v| + 1 + |j-u|)
    # But we also need that the intermediate paths don't go through the edge in a cycle
    # In a bipartite graph this is fine.
    
    # Precompute coverage: for each edge index, which far pairs it covers
    # Note: an edge covers a pair if the shortcut reduces dist to <= 4
    # But we need ALL pairs to be <= 4, so we need the union of coverages to cover all far pairs
    # AND the actual diameter must be <= 4 (which is stronger)
    
    # Let's use a different approach: branch and bound with fast validation
    
    def build_adj(added_indices):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for idx in added_indices:
            u, v = bipartite_candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_diameter(adj, limit=4):
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= limit:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        if dist[v] > limit:
                            # Still need to check connectivity, but diameter exceeded
                            pass
                        queue.append(v)
            if -1 in dist:
                return False
            if any(d > limit for d in dist if d != -1):
                return False
        return True
    
    def is_valid(added_indices):
        adj = build_adj(added_indices)
        return check_diameter(adj)
    
    # Fast check: for a set of added edges, verify diameter <= 4
    # Use BFS from each vertex but early terminate
    
    # Strategy: try k = 3, 4, 5, ... and find minimum valid set
    # Use branch-and-bound with pruning
    
    # Precompute: for each far pair, which edges can cover it
    # Edge (u,v) covers pair (i,j) if min(|i-u|+1+|j-v|, |i-v|+1+|j-u|) <= 4
    # This is a necessary condition (not sufficient for full diameter)
    
    edge_covers = [set() for _ in range(m)]
    for ei, (u, v) in enumerate(bipartite_candidates):
        for pi, (i, j) in enumerate(far_pairs):
            d1 = abs(i - u) + 1 + abs(j - v)
            d2 = abs(i - v) + 1 + abs(j - u)
            if min(d1, d2) <= 4:
                edge_covers[ei].add(pi)
    
    # For each far pair, which edges cover it
    pair_edges = [set() for _ in range(len(far_pairs))]
    for ei in range(m):
        for pi in edge_covers[ei]:
            pair_edges[pi].add(ei)
    
    # If any far pair has no covering edge, no solution exists (shouldn't happen)
    
    # Branch and bound: select edges to cover all far pairs
    # Lower bound: max over uncovered pairs of (number of uncovered pairs / max coverage per edge)
    # Better: use the fact that we need to cover all pairs
    
    best_solution = None
    best_count = 10  # current best from seed
    
    # Try to find solutions with k < best_count
    # Use recursive backtracking with pruning
    
    def solve():
        nonlocal best_solution, best_count
        deadline_local = deadline - 1  # safety margin
        
        # State: selected edges, remaining far pairs to cover
        selected = []
        remaining = set(range(len(far_pairs)))
        
        def lower_bound(remaining_pairs):
            if not remaining_pairs:
                return 0
            # Each new edge can cover at most max_cov pairs
            max_cov = 0
            for ei in range(m):
                if ei not in selected_set:
                    cov = len(edge_covers[ei] & remaining_pairs)
                    if cov > max_cov:
                        max_cov = cov
            if max_cov == 0:
                return float('inf')
            import math
            return math.ceil(len(remaining_pairs) / max_cov)
        
        def backtrack(remaining_pairs, selected_list, selected_set):
            nonlocal best_solution, best_count
            if time.time() > deadline_local:
                return
            if not remaining_pairs:
                if len(selected_list) < best_count:
                    best_count = len(selected_list)
                    best_solution = list(selected_list)
                return
            if len(selected_list) + lower_bound(remaining_pairs) >= best_count:
                return
            
            # Choose the most constrained pair (fewest covering edges)
            min_count = float('inf')
            chosen_pair = -1
            for pi in remaining_pairs:
                avail = len(pair_edges[pi] - selected_set)
                if avail == 0:
                    return  # dead end
                if avail < min_count:
                    min_count = avail
                    chosen_pair = pi
            
            # Try each edge that covers chosen_pair
            candidates = pair_edges[chosen_pair] - selected_set
            # Sort by coverage (most covering first)
            cand_list = sorted(candidates, key=lambda ei: -len(edge_covers[ei] & remaining_pairs))
            
            for ei in cand_list:
                if time.time() > deadline_local:
                    return
                if len(selected_list) + 1 >= best_count:
                    break
                new_remaining = remaining_pairs - edge_covers[ei]
                selected_list.append(ei)
                selected_set.add(ei)
                backtrack(new_remaining, selected_list, selected_set)
                selected_list.pop()
                selected_set.discard(ei)
        
        backtrack(remaining, selected, set())
    
    solve()
    
    # The set cover solution is necessary but not sufficient for diameter <= 4
    # Verify and if valid, use it. Otherwise, fall back to greedy + local search.
    
    if best_solution is not None and is_valid(best_solution):
        result = []
        for idx in best_solution:
            u, v = bipartite_candidates[idx]
            result.append([int(u), int(v)])
        return result
    
    # Fallback: greedy + local search
    def greedy_build():
        added = set()
        adj = build_adj(added)
        
        while True:
            if time.time() > deadline - 1:
                break
            
            # Count far pairs
            all_dists = []
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                all_dists.append(dist)
            
            far_count = sum(1 for i in range(n) for j in range(i+1, n) if all_dists[i][j] > 4)
            if far_count == 0:
                break
            
            best_edge = None
            best_far = far_count
            
            for u, v in bipartite_candidates:
                if (u, v) in added:
                    continue
                new_adj = [set(adj[i]) for i in range(n)]
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                new_far = 0
                for src in range(n):
                    dist = [-1] * n
                    dist[src] = 0
                    q = deque([src])
                    while q:
                        node = q.popleft()
                        for nb in new_adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                q.append(nb)
                    for j in range(src+1, n):
                        if dist[j] > 4:
                            new_far += 1
                
                if new_far < best_far:
                    best_far = new_far
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            added.add(best_edge)
            adj = build_adj(added)
        
        return added
    
    def local_search_reduce(added_edges):
        added = set(added_edges)
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            for edge in list(added):
                if time.time() > deadline - 1:
                    break
                new_set = added - {edge}
                if is_valid(new_set):
                    added = new_set
                    improved = True
                    break
        return added
    
    # Known good solution
    hub_solution = [(1, 11), (4, 11), (7, 11), (11, 15), (11, 18), (11, 21), (11, 23)]
    best = set(hub_solution)
    
    greedy_result = greedy_build()
    if greedy_result:
        greedy_result = local_search_reduce(greedy_result)
        if is_valid(greedy_result) and len(greedy_result) < len(best):
            best = greedy_result
    
    # Try hub-based
    for h in range(n):
        if time.time() > deadline - 1:
            break
        added = set()
        for v in range(n):
            if v == h:
                continue
            if abs(h - v) > 2 and (h % 2) != (v % 2):
                added.add((min(h,v), max(h,v)))
        if added and is_valid(added):
            added = local_search_reduce(added)
            if is_valid(added) and len(added) < len(best):
                best = added
    
    best = local_search_reduce(best)
    
    result = []
    for u, v in best:
        result.append([int(u), int(v)])
    
    return result
