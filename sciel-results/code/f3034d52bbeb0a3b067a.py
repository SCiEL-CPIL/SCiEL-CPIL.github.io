"""Decompose edges into cycles and single edges, minimizing total pieces."""
import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adjacency(edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle(adj, n, start=None):
    """Find a cycle using DFS. Returns list of vertices in cycle order, or None."""
    if start is None:
        start = 0
    color = {i: 0 for i in range(n)}  # 0=white, 1=gray, 2=black
    parent = {i: -1 for i in range(n)}
    
    def dfs(u):
        color[u] = 1
        for v in sorted(adj[u]):
            if v == parent[u]:
                continue
            if color[v] == 0:
                parent[v] = u
                result = dfs(v)
                if result:
                    return result
            elif color[v] == 1:
                # Found back edge u->v, extract cycle
                cycle = [v]
                curr = u
                while curr != v:
                    cycle.append(curr)
                    curr = parent[curr]
                cycle.reverse()
                return cycle
        color[u] = 2
        return None
    
    result = dfs(start)
    if result:
        return result
    # Try other starting points
    for s in range(1, n):
        if color[s] == 0:
            # Reset for new DFS
            color = {i: 0 for i in range(n)}
            parent = {i: -1 for i in range(n)}
            result = dfs(s)
            if result:
                return result
    return None

def _find_path_between(adj, n, u, v, max_len=50):
    """Find a simple path from u to v using BFS. Returns path or None."""
    from collections import deque
    if u == v:
        return None
    if v not in adj[u]:
        return None
    # BFS from u to v
    parent = {u: -1}
    queue = deque([u])
    while queue:
        curr = queue.popleft()
        if curr == v:
            # Reconstruct path
            path = [v]
            while path[-1] != u:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        for nb in sorted(adj[curr]):
            if nb not in parent:
                parent[nb] = curr
                queue.append(nb)
    return None

def _decompose_graph(edges, n, deadline):
    """Decompose edges into cycles and single edges."""
    edge_set = set(edges)
    adj = _build_adjacency(edges, n)
    pieces = []
    
    # Phase 1: Extract cycles repeatedly
    while time.time() < deadline:
        if not edge_set:
            break
        # Find a cycle
        cycle = None
        for start in range(n):
            if adj[start]:  # only if vertex has edges
                cycle = _find_cycle(adj, n, start)
                if cycle:
                    break
        if not cycle:
            break
        
        # Extract cycle edges
        cycle_edges = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            cycle_edges.add(e)
        
        # Verify all edges exist
        valid = all(e in edge_set for e in cycle_edges)
        if not valid:
            # Fallback: try to find a valid cycle more carefully
            break
        
        # Remove cycle edges
        for e in cycle_edges:
            edge_set.discard(e)
            adj[e[0]].discard(e[1])
            adj[e[1]].discard(e[0])
        
        pieces.append(cycle)
    
    # Phase 2: Try to close paths into cycles
    while time.time() < deadline:
        if not edge_set:
            break
        found = False
        # For each remaining edge (u,v), try to find a path from u to v
        for e in list(edge_set):
            u, v = e
            if not adj[u] or not adj[v]:
                continue
            # Find a path from u to v that doesn't use edge (u,v) directly
            # Temporarily remove edge
            adj[u].discard(v)
            adj[v].discard(u)
            path = _find_path_between(adj, n, u, v)
            adj[u].add(v)
            adj[v].add(u)
            
            if path and len(path) >= 2:
                # Path is u = path[0], path[1], ..., path[-1] = v
                # Cycle is: path + closing edge (v,u)
                cycle = path  # already starts at u and ends at v
                # Verify all edges in path exist
                valid = True
                cycle_edges = set()
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    ce = tuple(sorted((a, b)))
                    if ce not in edge_set:
                        valid = False
                        break
                    cycle_edges.add(ce)
                # Add closing edge
                ce = tuple(sorted((u, v)))
                if ce not in edge_set:
                    valid = False
                else:
                    cycle_edges.add(ce)
                
                if valid:
                    # Remove all cycle edges
                    for ce in cycle_edges:
                        edge_set.discard(ce)
                        adj[ce[0]].discard(ce[1])
                        adj[ce[1]].discard(ce[0])
                    pieces.append(cycle)
                    found = True
                    break
        
        if not found:
            break
    
    # Phase 3: Remaining edges become single-edge pieces
    for e in sorted(edge_set):
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        if time.time() > deadline:
            # Emergency: return all edges as single pieces
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(n, p, seed)
        pieces = _decompose_graph(edges, n, deadline)
        results.append(pieces)
    return results
