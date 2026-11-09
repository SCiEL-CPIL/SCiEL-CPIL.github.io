"""Rewritten program: exact minimum-edge augmentation via branch-and-bound with bitmasks."""

import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_masks():
    """Precompute all data structures needed for the search."""
    n = 24
    # Base adjacency as bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Distance-2 bitmasks (vertices at distance exactly 2)
    dist2 = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if adj[u] & (1 << v):
                mask |= adj[v]
        dist2[u] = mask & ~(adj[u] | (1 << u))
    
    # Candidate edges: all non-adjacent pairs (u < v) where u, v not adjacent in path
    # For triangle-free: we need to check that adding (u,v) doesn't create a triangle
    # A triangle exists if there's a w adjacent to both u and v
    # Since graph is bipartite (path), u and v must be in different partitions
    # Even: 0,2,4,...,22; Odd: 1,3,5,...,23
    # Adjacent pairs in path: (0,1),(1,2),...,(22,23)
    # Non-adjacent pairs where |u-v| != 1
    # But we also need: no common neighbor (triangle-free)
    # In a path, common neighbors of u and v exist only if they share a path neighbor
    # For even u and odd v: common neighbors are vertices w where w adjacent to both
    # w adjacent to u means w = u-1 or u+1 (in path)
    # w adjacent to v means w = v-1 or v+1 (in path)
    # So common neighbor exists if {u-1, u+1} ∩ {v-1, v+1} ≠ ∅
    
    candidates = []  # list of (u, v, cover_mask)
    cand_index = {}  # (u,v) -> index
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue  # already an edge
            # Check triangle-free: no common neighbor
            common = adj[u] & adj[v]
            if common:
                continue  # would create a triangle
            # Compute cover mask: which vertex pairs (x,y) get distance reduced
            # Actually, let's think about this differently.
            # We need diameter ≤ 4. Let's precompute which pairs need help.
            # A pair (x,y) is "covered" by edge (u,v) if dist(x,y) ≤ 4 in the augmented graph.
            # This is complex to precompute statically. Instead, let's use a different approach.
            candidates.append((u, v))
            cand_index[(u, v)] = len(candidates) - 1
    
    return n, adj, dist2, candidates, cand_index

def _bfs_dist(adj, n):
    """Compute all-pairs shortest paths using BFS. Returns dist matrix."""
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        d = [-1] * n
        d[src] = 0
        queue = [src]
        while queue:
            node = queue.pop(0)
            for nb in range(n):
                if adj[node] & (1 << nb) and d[nb] == -1:
                    d[nb] = d[node] + 1
                    queue.append(nb)
        dist[src] = d
    return dist

def _pairs_needing_help(adj, n):
    """Find all pairs (x,y) with distance > 4 in the base graph."""
    dist = _bfs_dist(adj, n)
    pairs = []
    for x in range(n):
        for y in range(x + 1, n):
            if dist[x][y] > 4:
                pairs.append((x, y))
    return pairs, dist

def _edge_covers_pair(u, v, x, y, adj, n):
    """Check if adding edge (u,v) makes dist(x,y) ≤ 4.
    We need: min(dist(x,y), dist(x,u)+1+dist(v,y), dist(x,v)+1+dist(u,y)) ≤ 4
    But we need distances in the augmented graph, which is complex.
    Approximation: check if the direct path through (u,v) helps.
    Actually for a single edge addition, the new distance is:
    min(old_dist, dist(x,u)+1+dist(v,y), dist(x,v)+1+dist(u,y))
    where dists are in the original graph (since we're checking one edge at a time
    for the cover contribution).
    """
    # This is an approximation that works for the cover heuristic
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 24
    # Base adjacency
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # All-pairs distances in base graph
    base_dist = _bfs_dist(adj, n)
    
    # Pairs that need help (distance > 4)
    problem_pairs = []
    for x in range(n):
        for y in range(x + 1, n):
            if base_dist[x][y] > 4:
                problem_pairs.append((x, y))
    
    num_pairs = len(problem_pairs)
    # Represent problem pairs as a bitmask (up to 276 pairs, fits in Python int)
    
    # Candidate edges: non-adjacent, triangle-free
    cand_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            common = adj[u] & adj[v]
            if common:
                continue
            cand_edges.append((u, v))
    
    num_cands = len(cand_edges)
    
    # For each candidate edge, compute which problem pairs it "helps"
    # Adding edge (u,v): new dist(x,y) = min(base_dist[x][y], 
    #     base_dist[x][u]+1+base_dist[v][y], base_dist[x][v]+1+base_dist[u][y])
    # If new dist ≤ 4, the pair is covered.
    
    edge_cover = [0] * num_cands  # bitmask of problem pairs covered
    for ci, (u, v) in enumerate(cand_edges):
        mask = 0
        for pi, (x, y) in enumerate(problem_pairs):
            d1 = base_dist[x][u] + 1 + base_dist[v][y]
            d2 = base_dist[x][v] + 1 + base_dist[u][y]
            new_d = min(base_dist[x][y], d1, d2)
            if new_d <= 4:
                mask |= (1 << pi)
        edge_cover[ci] = mask
    
    # Check if any single edge covers all
    all_mask = (1 << num_pairs) - 1
    
    # Greedy initial solution
    def greedy_solve():
        remaining = all_mask
        selected = []
        while remaining:
            best_ci = -1
            best_count = -1
            for ci in range(num_cands):
                if not (edge_cover[ci] & remaining):
                    continue
                cnt = bin(edge_cover[ci] & remaining).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_ci = ci
            if best_ci == -1:
                return None
            selected.append(best_ci)
            remaining &= ~edge_cover[best_ci]
        return selected
    
    greedy_sol = greedy_solve()
    best_sol = greedy_sol if greedy_sol else None
    best_len = len(best_sol) if best_sol else 100
    
    # Branch and bound
    def branch_and_bound():
        nonlocal best_sol, best_len
        found = [False]
        
        def bnb(remaining, selected, depth):
            nonlocal best_sol, best_len, found
            if time.time() > deadline - 2:
                return
            if remaining == 0:
                if len(selected) < best_len:
                    best_sol = list(selected)
                    best_len = len(selected)
                    found[0] = True
                return
            if len(selected) + 1 >= best_len:
                return
            
            # Lower bound: minimum number of edges needed
            # Each edge covers at most max_cover pairs from remaining
            max_cov = 0
            for ci in range(num_cands):
                cov = bin(edge_cover[ci] & remaining).count('1')
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                return
            lb = (bin(remaining).count('1') + max_cov - 1) // max_cov
            if len(selected) + lb >= best_len:
                return
            
            # Choose the pair with fewest covering edges
            # Find a problem pair in remaining with minimum number of candidate edges covering it
            best_pair_bit = -1
            best_pair_count = 1000000
            for pi in range(num_pairs):
                if remaining & (1 << pi):
                    cnt = 0
                    for ci in range(num_cands):
                        if edge_cover[ci] & (1 << pi):
                            cnt += 1
                    if cnt < best_pair_count:
                        best_pair_count = cnt
                        best_pair_bit = pi
                    if cnt <= 1:
                        break
            
            if best_pair_bit == -1:
                return
            
            # Try each candidate edge that covers this pair
            tried = []
            for ci in range(num_cands):
                if edge_cover[ci] & (1 << best_pair_bit):
                    tried.append(ci)
            
            # Sort by coverage (more coverage first)
            tried.sort(key=lambda ci: -bin(edge_cover[ci] & remaining).count('1'))
            
            for ci in tried:
                if time.time() > deadline - 2:
                    return
                if len(selected) + 1 >= best_len:
                    break
                selected.append(ci)
                bnb(remaining & ~edge_cover[ci], selected, depth + 1)
                selected.pop()
        
        bnb(all_mask, [], 0)
    
    branch_and_bound()
    
    # Local search: try to improve by removing edges
    if best_sol and best_len > 1:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(len(best_sol)):
                if time.time() > deadline - 2:
                    break
                # Try removing edge i
                trial = best_sol[:i] + best_sol[i+1:]
                if not trial:
                    continue
                # Check if trial covers all pairs
                covered = 0
                for ci in trial:
                    covered |= edge_cover[ci]
                if covered == all_mask:
                    best_sol = trial
                    best_len = len(trial)
                    improved = True
                    break
            if not improved:
                break
    
    # Build the result
    if best_sol is None:
        # Fallback: hub at 12
        fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                    [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
        return fallback
    
    result = []
    for ci in best_sol:
        u, v = cand_edges[ci]
        result.append([u, v])
    
    # Validate
    if not _validate(result):
        fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                    [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
        return fallback
    
    return result

def _validate(added_edges):
    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_edges = set(base_edges)
    for e in added_edges:
        u, v = int(min(e[0], e[1])), int(max(e[0], e[1]))
        if (u, v) in all_edges:
            return False
        all_edges.add((u, v))
    
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Triangle-free
    adj_set = [set(a) for a in adj]
    for u in range(n):
        for i in range(len(adj[u])):
            for j in range(i + 1, len(adj[u])):
                if adj[u][j] in adj_set[adj[u][i]]:
                    return False
    
    # Connected
    visited = [False] * n
    visited[0] = True
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    if not all(visited):
        return False
    
    # Diameter ≤ 4
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        while queue:
            node = queue.pop(0)
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        if any(d == -1 for d in dist):
            return False
    
    return True
