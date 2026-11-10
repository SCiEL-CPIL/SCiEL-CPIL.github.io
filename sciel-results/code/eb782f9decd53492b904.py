"""Hexagon-free graph on 40 vertices: greedy edge addition with C6 check."""
import time
import random

DEADLINE = 110.0  # safety margin before 120s limit


def _has_c6(adj, u, v, n):
    """Check if adding edge (u,v) to graph with adjacency sets adj creates a C6."""
    # A C6 through new edge (u,v) means a path of length 5 from u to v
    # that avoids using edge (u,v) itself.
    # BFS from u up to depth 5, tracking paths, looking for v at depth 5
    # with exactly 5 edges (so 6 vertices: u, a, b, c, d, v)
    # Actually we need a simple path of length 5: u-a-b-c-d-v
    # Use BFS with parent tracking up to depth 5 from u, see if we reach v at depth 5.
    # But we must ensure the path is simple (no repeated vertices).
    
    # Better: BFS from u, depth 0..5. At depth 5 check if v is reachable.
    # Track (vertex, depth, path_set) is too expensive.
    # Instead: BFS layers. At each layer, track which vertices are reachable.
    # For C6 detection: we need a path u -> x1 -> x2 -> x3 -> x4 -> v of length 5.
    # This is equivalent to: there exist x1 in N(u)\{v}, x4 in N(v)\{u},
    # and a path of length 3 from x1 to x4 avoiding u,v and being simple.
    
    # Simpler approach: BFS from u up to depth 5, but we need simple paths.
    # For n=40, this is manageable with careful implementation.
    
    # Alternative: check if there's a path of length exactly 5 from u to v
    # that doesn't use edge (u,v). We can do BFS from u and track the shortest
    # path, but we need *any* path of length 5, not just shortest.
    
    # Most reliable: enumerate all simple paths of length 5 from u to v.
    # Depth-first search from u, depth 0 to 5, looking for v at depth 5.
    # n=40, branching factor ~ degree. For sparse-ish graphs this is ok.
    
    # Let's do iterative deepening / DFS with visited set.
    # To keep it fast, use a set for visited.
    
    def dfs(node, depth, visited):
        if depth == 5:
            return node == v
        for nb in adj[node]:
            if nb == u and depth == 0:
                continue  # don't go back to u
            if nb == v and depth < 4:
                continue  # can't use v before the end
            if nb in visited:
                continue
            visited.add(nb)
            if dfs(nb, depth + 1, visited):
                visited.discard(nb)
                return True
            visited.discard(nb)
        return False
    
    visited = {u}
    return dfs(u, 0, visited)


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def construct():
    start_time = time.time()
    n = 40
    
    # Start with 8 disjoint K5's
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    
    adj = _build_adj(edges, n)
    
    # Generate all possible edges not already present
    existing = set()
    for u, v in edges:
        existing.add((min(u, v), max(u, v)))
    
    # Candidate edges: all pairs not already present
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing:
                candidates.append((u, v))
    
    # Shuffle for variety
    random.seed(42)
    random.shuffle(candidates)
    
    # Greedy: try to add each candidate if it doesn't create a C6
    for u, v in candidates:
        if time.time() > DEADLINE:
            break
        if _has_c6(adj, u, v, n):
            continue
        # Add the edge
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
    
    # Try a few more passes with different orderings if time permits
    remaining = []
    for u, v in candidates:
        if (u, v) not in existing and not any(e[0] == u and e[1] == v or e[0] == v and e[1] == u for e in edges):
            remaining.append((u, v))
    
    # Second pass with reversed order
    if time.time() < DEADLINE - 5:
        random.shuffle(remaining)
        for u, v in remaining:
            if time.time() > DEADLINE:
                break
            if (u, v) in existing:
                continue
            if _has_c6(adj, u, v, n):
                continue
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
    
    return edges
