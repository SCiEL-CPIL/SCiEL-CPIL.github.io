import time
import random

# EVOLVE-BLOCK-START

def _valid_edges(n, min_deg, min_girth, max_time):
    """
    Generate a valid edge list for a graph on vertices 0..n-1.
    Ensures:
      - simple, undirected
      - no loops
      - every vertex has degree >= min_deg
      - every simple cycle has length >= min_girth (i.e., girth constraint)
      - contains at least one cycle
    Returns list of [u, v] or None if no valid graph found within max_time.
    """
    start = time.time()
    deadline = start + max_time - 2.0  # safety margin

    # Strategy: build two disjoint K3,3 components
    # K3,3 on vertices 0..5: partitions [0,1,2] and [3,4,5]
    # K3,3 on vertices 6..11: partitions [6,7,8] and [9,10,11]
    # This gives min degree 3, girth 4, and cycle lengths only 4 and 6.
    # Distinct cycle lengths = {4, 6} -> count = 2.

    edges = []
    # First K3,3
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    # Second K3,3
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])

    # Validate
    if _check_valid(n, min_deg, min_girth, edges, deadline):
        return edges

    # Fallback: try random search for a valid graph
    # (unlikely to be needed, but include for robustness)
    for _ in range(200):
        if time.time() > deadline:
            break
        cand = _random_candidate(n, min_deg, deadline)
        if cand is not None and _check_valid(n, min_deg, min_girth, cand, deadline):
            return cand

    return None


def _check_valid(n, min_deg, min_girth, edges, deadline):
    """Check if edges form a valid graph satisfying all constraints."""
    if time.time() > deadline:
        return False

    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)

    # Check min degree
    for i in range(n):
        if len(adj[i]) < min_deg:
            return False

    # Check girth: no cycle shorter than min_girth
    # For min_girth=4, just check no triangles
    if min_girth >= 4:
        # Check no triangles
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False  # triangle found

    # Check contains a cycle
    if not _has_cycle(n, adj):
        return False

    return True


def _has_cycle(n, adj):
    """Check if graph has at least one cycle using BFS/DFS."""
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            if _dfs_has_cycle(i, -1, visited, adj):
                return True
    return False


def _dfs_has_cycle(node, parent, visited, adj):
    visited[node] = True
    for neighbor in adj[node]:
        if not visited[neighbor]:
            if _dfs_has_cycle(neighbor, node, visited, adj):
                return True
        elif neighbor != parent:
            return True
    return False


def _random_candidate(n, min_deg, deadline):
    """Generate a random candidate graph."""
    edges = []
    edge_set = set()
    adj = [set() for _ in range(n)]

    # First, ensure min degree by adding random edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)

    for u, v in all_pairs:
        if time.time() > deadline:
            return None
        if len(adj[u]) < min_deg or len(adj[v]) < min_deg:
            edges.append([u, v])
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)

    # Check if min degree satisfied
    for i in range(n):
        if len(adj[i]) < min_deg:
            return None

    return edges


def construct():
    n = 12
    min_deg = 3
    min_girth = 4
    max_time = 55.0  # well within 60s limit

    result = _valid_edges(n, min_deg, min_girth, max_time)
    if result is not None:
        return result

    # Absolute fallback: return the K3,3+K3,3 construction directly
    edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
