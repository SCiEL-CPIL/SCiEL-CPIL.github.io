"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict

# Hard deadline guard (evaluator kills at 150s; use 120s budget with safety margin)
_DEADLINE_SECONDS = 110.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(edges, n):
    """Build adjacency list and degree counts."""
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg


def _find_cycle(adj, deg, n):
    """
    Find a (preferably long) cycle in the current graph.
    Returns a list of vertices forming a cycle (length >= 3), or None.
    Strategy:
      1) For each high-degree start vertex, do a DFS that prefers unvisited
         high-degree neighbors; if we reach `start` with path length >= 3 we
         have a cycle.  We try several start vertices.
      2) Fallback: for each edge (u,v), BFS from v to u avoiding the direct
         edge to get a simple path, giving a cycle.
    """
    candidates = [v for v in range(n) if deg[v] >= 2]
    if not candidates:
        return None
    candidates.sort(key=lambda v: -deg[v])

    for start in candidates:
        if deg[start] < 2:
            continue
        # Iterative DFS with explicit stack of (node, path, next_neighbor_idx)
        # We want to find ANY cycle back to start; prefer longer by exploring
        # deeper first (DFS).  To bias toward longer cycles, we try to extend
        # the path before closing it.
        stack = [(start, [start], 0)]
        visited = {start}
        while stack:
            node, path, ni = stack.pop()
            neighbors = sorted(adj[node], key=lambda x: -deg[x])
            # Continue from where we left off
            for j in range(ni, len(neighbors)):
                nb = neighbors[j]
                if nb == start and len(path) >= 3:
                    return path
                if nb not in visited:
                    visited.add(nb)
                    # Push the "resume" frame for current node, then descend
                    stack.append((node, path, j + 1))
                    stack.append((nb, path + [nb], 0))
                    break
            # If we exhausted neighbors for this frame, just pop (implicit)

    # Fallback: edge-based BFS
    all_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                all_edges.add((u, v))
    if not all_edges:
        return None

    edge_list = list(all_edges)
    random.shuffle(edge_list)
    from collections import deque
    for u, v in edge_list:
        # BFS from v to u avoiding direct edge (u,v)
        queue = deque()
        queue.append((v, [v]))
        visited = {v}
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node], key=lambda x: -deg[x]):
                if nb == u and len(path) >= 2:
                    return path + [u]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
    return None


def _remove_cycle(cycle, adj, deg):
    """Remove edges of the cycle from the graph."""
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1


def _remove_edge(u, v, adj, deg):
    """Remove a single edge from the graph."""
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1


def _decompose_graph(edges, n, time_limit):
    """
    Decompose a graph into cycles and edges.
    Returns a list of pieces (each piece is either [u,v] for an edge or a list of >=3 vertices for a cycle).
    """
    adj, deg = _build_adjacency(edges, n)
    pieces = []
    start_time = time.time()

    # Greedily find cycles while possible
    while True:
        if time.time() - start_time > time_limit:
            break

        # Check if any edges remain
        remaining_edges = 0
        for u in range(n):
            remaining_edges += len(adj[u])
        remaining_edges //= 2

        if remaining_edges == 0:
            break

        # Try to find a cycle
        cycle = _find_cycle(adj, deg, n)
        if cycle is not None:
            pieces.append(cycle)
            _remove_cycle(cycle, adj, deg)
        else:
            # No cycle found; take an edge
            # Find any remaining edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            _remove_edge(u, v, adj, deg)

    # Add any remaining edges as individual pieces
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])

    # Post-processing: try to merge consecutive edge pieces into cycles
    # If we have a set of single-edge pieces, check if any 3+ form a cycle
    # in the original graph.  This is a heuristic local improvement.
    pieces = _merge_edge_pieces(pieces, edges, n)

    return pieces


def _merge_edge_pieces(pieces, original_edges, n):
    """
    Try to replace some single-edge pieces with cycles, reducing total count.
    A cycle of length k replaces k edge-pieces with 1 cycle-piece (saving k-1).
    We look for cycles in the subgraph induced by edges that are currently
    single-edge pieces.
    """
    # Identify which edges are currently single-edge pieces
    edge_piece_set = set()
    for p in pieces:
        if len(p) == 2:
            edge_piece_set.add(tuple(sorted(p)))

    if len(edge_piece_set) < 3:
        return pieces

    # Build adjacency for just the edge-piece subgraph
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edge_piece_set:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    # Find cycles in this subgraph and merge them
    new_pieces = []
    remaining = set(edge_piece_set)

    # Try to find cycles greedily in the edge-piece subgraph
    while True:
        # Find a cycle in the remaining edge-piece subgraph
        cycle = None
        for u in range(n):
            if deg[u] >= 2:
                # BFS from u to find a cycle back to u
                from collections import deque
                queue = deque()
                queue.append((u, [u]))
                visited = {u}
                while queue and cycle is None:
                    node, path = queue.popleft()
                    for nb in sorted(adj[node], key=lambda x: -deg[x]):
                        if nb == u and len(path) >= 3:
                            cycle = path
                            break
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))
                if cycle:
                    break
        if not cycle:
            break

        # Verify all edges in the cycle are in remaining
        cycle_edges = set()
        ok = True
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in remaining:
                ok = False
                break
            cycle_edges.add(e)
        if not ok:
            break

        # Replace these edge pieces with a cycle
        new_pieces.append(cycle)
        for e in cycle_edges:
            remaining.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1

    # Add remaining edge pieces
    for e in remaining:
        new_pieces.append(list(e))

    return new_pieces


def _decompose_with_restart(edges, n, time_limit, num_restarts=5):
    """
    Try multiple random restarts with different orderings to find a better decomposition.
    """
    best_pieces = None
    best_count = float('inf')
    per_restart_time = time_limit / max(num_restarts, 1)

    for restart in range(num_restarts):
        if time.time() > time_limit:
            break

        pieces = _decompose_graph(edges, n, per_restart_time)
        count = len(pieces)

        if count < best_count:
            best_count = count
            best_pieces = pieces

    if best_pieces is None:
        # Fallback: all edges as individual pieces
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            # Not enough time; return all edges as pieces for remaining graphs
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)

        # Allocate time for this graph, leaving buffer for later graphs
        graphs_left = len(specs) - i
        graph_time = remaining * 0.85 / graphs_left * graphs_left * 0.3
        # Simplify: give each graph a proportional share
        graph_time = remaining * 0.85 / max(graphs_left, 1)

        # More restarts for smaller graphs, fewer for larger
        num_restarts = max(2, min(5, int(30 / max(n, 1)) * 2))

        pieces = _decompose_with_restart(edges, n, graph_time, num_restarts=num_restarts)
        results.append(pieces)

    return results
