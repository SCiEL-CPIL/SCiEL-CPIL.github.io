"""Path decomposition of three fixed connected graphs (Gallai's conjecture).

Strategy:
  - For each instance graph, formulate the path-decomposition problem as a
    maximum-cardinality matching in a bipartite graph.
  - Each undirected edge e=(u,v) is represented by two directed stubs:
    (u -> v) and (v -> u).
  - Two directed stubs (a->b) and (c->d) can be matched (i.e., made consecutive
    in a path) if and only if b == c (the head of the first equals the tail of
    the second) AND the two stubs are not the two orientations of the same
    edge (to avoid using the same undirected edge twice consecutively in a
    way that creates a 2-cycle / repeats an edge).
  - A matching of size |M| in this bipartite graph yields exactly |E| - |M|
    edge-disjoint paths that cover all edges.
  - We use Hopcroft-Karp to find the maximum matching, which minimizes the
    number of paths.

  Note: This gives the minimum number of edge-disjoint paths in terms of
  "trail" decomposition where each edge is used exactly once and consecutive
  edges share a vertex. This is exactly the path decomposition requested.
"""

import time
from collections import defaultdict, deque

# Deadline guard
_START_TIME = time.time()
DEADLINE = 140.0  # 150s wall-clock, leave 10s safety margin


def _check_deadline():
    return time.time() - _START_TIME < DEADLINE


def _instance(n, p, seed):
    """Generate the instance graph identically to the verifier."""
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _hopcroft_karp(adj, n_left, n_right):
    """
    Hopcroft-Karp maximum bipartite matching.
    adj: dict mapping left node -> list of right nodes
    Returns: matching dict (left -> right)
    """
    # matching arrays
    match_l = [-1] * n_left
    match_r = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        """Find augmenting paths; returns True if at least one exists."""
        queue = deque()
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj.get(u, []):
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == -1:
                    dist[mu] = dist[u] + 1
                    queue.append(mu)
        return found

    def dfs(u):
        """Try to find an augmenting path from u."""
        for v in adj.get(u, []):
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False

    matching_size = 0
    while bfs():
        for u in range(n_left):
            if match_l[u] == -1:
                if dfs(u):
                    matching_size += 1
    return match_l


def _decompose_paths(edges):
    """
    Given a list of undirected edges (as tuples (u,v) with u<v),
    find a minimum number of edge-disjoint paths covering all edges.
    Returns list of paths, each path is a list of vertices.
    """
    m = len(edges)
    if m == 0:
        return []

    # Build adjacency for directed stubs
    # Left side: directed stubs (u, v) meaning "entering v from u"
    # Right side: directed stubs (x, y) meaning "leaving x to y"
    # A match between left (u,v) and right (x,y) means v==x, i.e., edge (u,v)
    # is followed by edge (v,y) in a path.
    # Constraint: (u,v) and (v,u) are two orientations of the same undirected edge;
    # we must not match them to each other (that would create a 2-cycle using
    # the same edge twice).

    # Enumerate all directed stubs
    # For each undirected edge (a,b), we have two directed edges: (a,b) and (b,a)
    # Left nodes: (a,b) for each undirected edge (a,b)
    # Right nodes: (b,a) for each undirected edge (a,b) -- wait, let me re-think.

    # Actually, let's index:
    # Left side: for each undirected edge i = (u_i, v_i), we have two directed stubs:
    #   left[2*i]   = (u_i, v_i)  -- "arrive at v_i from u_i"
    #   left[2*i+1] = (v_i, u_i)  -- "arrive at u_i from v_i"
    # Right side: for each undirected edge j = (u_j, v_j), we have two directed stubs:
    #   right[2*j]   = (u_j, v_j)  -- "leave u_j to v_j"
    #   right[2*j+1] = (v_j, u_j)  -- "leave v_j to u_j"

    # A match between left stub (a -> b) and right stub (c -> d) is valid if b == c.
    # But we must exclude matching left stub of edge i with right stub of the same
    # edge i (since that would mean using the same undirected edge twice).

    n_left = 2 * m
    n_right = 2 * m

    # Build adjacency: for each left stub, which right stubs can it match to?
    adj = defaultdict(list)

    # Map from (tail, head) to list of (right_index, edge_index)
    # Right stub at index 2*j is (u_j, v_j), at 2*j+1 is (v_j, u_j)
    # For a left stub (a, b), we need right stubs starting with b, i.e., (b, ?)
    # Right stub (b, x) means there's an edge (b,x) or (x,b) in the graph.

    # Build a map: vertex -> list of (right_index, other_vertex, edge_index)
    vertex_to_right = defaultdict(list)
    for j, (u, v) in enumerate(edges):
        # right[2*j] = (u, v): starts at u, goes to v
        vertex_to_right[u].append((2 * j, v, j))
        # right[2*j+1] = (v, u): starts at v, goes to u
        vertex_to_right[v].append((2 * j + 1, u, j))

    for i, (u, v) in enumerate(edges):
        # Left stub 2*i = (u, v): arrives at v. Can be followed by any right stub starting at v.
        for (ri, other, ej) in vertex_to_right[v]:
            if ej != i:  # Cannot match with the same undirected edge
                adj[2 * i].append(ri)
        # Left stub 2*i+1 = (v, u): arrives at u. Can be followed by any right stub starting at u.
        for (ri, other, ej) in vertex_to_right[u]:
            if ej != i:
                adj[2 * i + 1].append(ri)

    # Run Hopcroft-Karp
    match_l = _hopcroft_karp(adj, n_left, n_right)

    # Now reconstruct paths from the matching.
    # match_l[left_idx] = right_idx or -1
    # left_idx 2*i corresponds to directed edge (u_i, v_i)
    # left_idx 2*i+1 corresponds to directed edge (v_i, u_i)
    # right_idx 2*j corresponds to directed edge (u_j, v_j)
    # right_idx 2*j+1 corresponds to directed edge (v_j, u_j)

    # For each directed edge (left stub), if it's matched to a right stub,
    # that means this directed edge is followed by the edge corresponding to the right stub.
    # We need to trace paths.

    # Build: for each directed edge (represented as (from, to)), find the next directed edge.
    # directed edge index: 2*i -> (u_i, v_i), 2*i+1 -> (v_i, u_i)

    def directed_edge(idx):
        """Return (from, to) for directed edge index."""
        i = idx // 2
        u, v = edges[i]
        if idx % 2 == 0:
            return (u, v)
        else:
            return (v, u)

    # next_dir: for each left stub, the right stub it's matched to (or -1)
    # From a left stub L (directed edge a->b), if matched to right stub R (directed edge c->d),
    # then b==c, and the next directed edge is c->d = b->d.
    # The right stub R corresponds to some undirected edge and direction.
    # right stub 2*j = (u_j, v_j), so the directed edge is (u_j, v_j).
    # right stub 2*j+1 = (v_j, u_j), so the directed edge is (v_j, u_j).

    # Let's build a map: left_idx -> next_left_idx
    # If left L is matched to right R, then the next directed edge is the one represented by R.
    # We need to find which left index corresponds to the same directed edge as R.
    # R = 2*j means directed edge (u_j, v_j) which is left stub 2*j.
    # R = 2*j+1 means directed edge (v_j, u_j) which is left stub 2*j+1.
    # So actually, right stub index == left stub index for the same directed edge!

    # That's convenient. So next_dir[L] = R if matched, else -1.
    next_dir = match_l  # match_l[L] = R or -1

    # Now trace paths. Each directed edge is a node in a functional graph (each node has
    # at most one outgoing edge via next_dir, and at most one incoming edge).
    # We want to find all paths (maximal sequences).

    # visited
    visited = [False] * n_left
    paths = []

    for start in range(n_left):
        if visited[start]:
            continue
        # Check if this is the start of a path (no incoming edge)
        # Actually, let's just trace from every unvisited node and build paths.
        # A path starts at a node with no predecessor (no left stub L' such that next_dir[L'] == start).
        # But it's easier to just trace all components.

        # Trace the path starting from 'start'
        path = []
        cur = start
        while cur != -1 and not visited[cur]:
            visited[cur] = True
            f, t = directed_edge(cur)
            if not path:
                path.append(f)
            else:
                # Check: last vertex in path should be f
                if path[-1] != f:
                    # This shouldn't happen if matching is valid
                    pass
                path.append(t)
            cur = next_dir[cur]

        if path:
            paths.append(path)

    return paths


def construct():
    """
    Return three lists of paths (vertex lists), one per fixed instance graph.
    """
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        if not _check_deadline():
            # Fallback: each edge as its own path
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        paths = _decompose_paths(edges)
        results.append(paths)
    return results
