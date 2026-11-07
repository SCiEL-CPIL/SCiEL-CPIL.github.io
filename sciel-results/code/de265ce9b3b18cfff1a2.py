"""Structural construction for triangle-free diameter-4 augmentation."""
# EVOLVE-BLOCK-START
import time

P = {
    'n': 24,
    'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
              [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
              [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
    'diameter': 4
}

N = P['n']
BASE_EDGES = set()
BASE_ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    BASE_EDGES.add((u, v))
    BASE_EDGES.add((v, u))
    BASE_ADJ[u].add(v)
    BASE_ADJ[v].add(u)


def _bfs_diameter(adj):
    """Return diameter of graph, or -1 if disconnected."""
    max_d = 0
    for src in range(N):
        dist = [-1] * N
        dist[src] = 0
        queue = [src]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_d:
                        max_d = dist[v]
                    queue.append(v)
        for i in range(N):
            if dist[i] == -1:
                return -1
    return max_d


def _verify(added):
    """Check triangle-free, connected, diameter <= 4."""
    adj = [set() for _ in range(N)]
    for u in range(N):
        adj[u] = set(BASE_ADJ[u])
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    # Triangle-free check
    for u in range(N):
        for v in adj[u]:
            if v > u:
                if adj[u] & adj[v]:
                    return False
    d = _bfs_diameter(adj)
    return d != -1 and d <= 4


def _build_adj(added):
    adj = [set() for _ in range(N)]
    for u in range(N):
        adj[u] = set(BASE_ADJ[u])
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _can_add(adj, u, v):
    """Check if adding (u,v) keeps triangle-free and doesn't duplicate."""
    if (u, v) in BASE_EDGES or (v, u) in BASE_EDGES:
        return False
    if v in adj[u]:
        return False
    return not (adj[u] & adj[v])


def _compute_buckets(adj):
    """For each vertex, compute which distance buckets 1-4 are covered by neighbors."""
    buckets = [set() for _ in range(N)]
    for u in range(N):
        for v in adj[u]:
            d = abs(u - v)
            if 1 <= d <= 4:
                buckets[u].add(d)
    return buckets


def _find_missing(adj):
    """Find (vertex, distance) pairs that are not covered."""
    buckets = _compute_buckets(adj)
    missing = []
    for u in range(N):
        for d in range(1, 5):
            if d not in buckets[u]:
                missing.append((u, d))
    return missing


def _candidate_edges_for_missing(u, d, adj):
    """Find candidate edges (u, v) where v is at distance d from u on the path,
    and adding it is valid."""
    candidates = []
    for v in range(N):
        if abs(u - v) != d:
            continue
        if _can_add(adj, u, v):
            candidates.append(v)
    return candidates


def _edge_coverage(u, v, adj):
    """Count how many (vertex, distance) pairs this edge covers."""
    count = 0
    for x in (u, v):
        d = abs(u - v)
        if 1 <= d <= 4:
            # Check if this distance was missing for x
            if d not in {abs(x - y) for y in adj[x] if 1 <= abs(x - y) <= 4}:
                count += 1
    return count


def _greedy_construct():
    """Greedy: repeatedly add the edge that covers the most missing (vertex, distance) pairs."""
    adj = _build_adj([])
    added = []
    
    for iteration in range(200):
        missing = _find_missing(adj)
        if not missing:
            break
        
        # Find the edge that covers the most missing pairs
        best_gain = 0
        best_edge = None
        
        # For each missing (u, d), find candidate edges
        for u, d in missing:
            candidates = _candidate_edges_for_missing(u, d, adj)
            for v in candidates:
                # Count total coverage gain of adding (u, v)
                gain = 0
                for x in (u, v):
                    dist = abs(u - v)
                    if 1 <= dist <= 4:
                        # Check if dist is missing for x
                        existing_dists = set()
                        for y in adj[x]:
                            dd = abs(x - y)
                            if 1 <= dd <= 4:
                                existing_dists.add(dd)
                        if dist not in existing_dists:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
    
    return added


def _try_remove(added):
    """Try to remove edges while maintaining validity."""
    improved = True
    while improved:
        improved = False
        for i in range(len(added)):
            test = added[:i] + added[i + 1:]
            if _verify(test):
                added = test
                improved = True
                break
    return added


def construct():
    start = time.time()
    deadline = start + 50
    
    added = _greedy_construct()
    
    # Verify
    if not _verify(added):
        # Fallback: try a different construction
        added = _fallback_construct()
        if not _verify(added):
            return [[0, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 23]]
    
    # Try to minimize
    added = _try_remove(added)
    
    if time.time() < deadline:
        # Try local search for further reduction
        added = _local_minimize(added, deadline)
    
    res = [[int(u), int(v)] for u, v in added]
    if _verify(res):
        return res
    return res


def _fallback_construct():
    """Alternative construction: add edges at fixed intervals."""
    adj = _build_adj([])
    added = []
    # Add edges to create a structure with diameter <= 4
    # Strategy: connect vertices that are 4 apart where possible
    for u in range(0, N, 4):
        for v in range(u + 1, N):
            if abs(u - v) == 4 and _can_add(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                added.append([u, v])
    
    # Check if diameter is ok, if not add more
    while _bfs_diameter(adj) > 4 and len(added) < 30:
        missing = _find_missing(adj)
        if not missing:
            break
        best_gain = 0
        best_edge = None
        for u, d in missing:
            candidates = _candidate_edges_for_missing(u, d, adj)
            for v in candidates:
                gain = 0
                for x in (u, v):
                    dist = abs(u - v)
                    if 1 <= dist <= 4:
                        existing_dists = set()
                        for y in adj[x]:
                            dd = abs(x - y)
                            if 1 <= dd <= 4:
                                existing_dists.add(dd)
                        if dist not in existing_dists:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (u, v)
        if best_edge is None:
            break
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
    
    return added


def _local_minimize(added, deadline):
    """Local search to reduce edge count."""
    current = [list(e) for e in added]
    best = [list(e) for e in added]
    
    while time.time() < deadline:
        improved = False
        # Try removing each edge
        for i in range(len(current)):
            if time.time() > deadline:
                return best
            test = current[:i] + current[i + 1:]
            if _verify(test):
                current = test
                best = [list(e) for e in current]
                improved = True
                break
        if not improved:
            # Try swapping: remove one, add another
            if len(current) > 0:
                for i in range(len(current)):
                    if time.time() > deadline:
                        return best
                    adj = _build_adj(current)
                    u1, v1 = current[i]
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    test_removed = current[:i] + current[i + 1:]
                    
                    # Try adding a different edge
                    for u2 in range(N):
                        for v2 in range(u2 + 1, N):
                            if time.time() > deadline:
                                return best
                            if (u2, v2) in BASE_EDGES or (v2, u2) in BASE_EDGES:
                                continue
                            if v2 in adj[u2]:
                                continue
                            if adj[u2] & adj[v2]:
                                continue
                            test = test_removed + [[u2, v2]]
                            if len(test) < len(best) and _verify(test):
                                current = test
                                best = [list(e) for e in current]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
    
    return best

# EVOLVE-BLOCK-END
