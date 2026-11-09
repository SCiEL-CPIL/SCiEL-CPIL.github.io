"""Decompose each instance graph into edge-disjoint cycles and edges, minimizing piece count."""
import time
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycle_dfs(adj, start):
    """Find a simple cycle via DFS from start. Returns list of vertices (cycle) or None."""
    parent = {start: None}
    path = [start]
    
    def dfs(node):
        for nb in sorted(adj.get(node, [])):
            if nb == parent.get(node):
                continue
            if nb in parent:
                if nb in path:
                    idx = path.index(nb)
                    return path[idx:]
            else:
                parent[nb] = node
                path.append(nb)
                res = dfs(nb)
                if res:
                    return res
                path.pop()
        return None
    
    return dfs(start)


def _find_cycle_from(adj, start):
    """Find a cycle in the subgraph reachable, preferring to find one through start."""
    # Use iterative DFS to avoid recursion limits
    stack = [(start, iter(sorted(adj.get(start, []))), [start])]
    parent = {start: None}
    while stack:
        node, it, path = stack[-1]
        found = False
        for nb in it:
            if nb == parent.get(node):
                continue
            if nb in path:
                idx = path.index(nb)
                return path[idx:]
            if nb not in parent:
                parent[nb] = node
                stack.append((nb, iter(sorted(adj.get(nb, []))), path + [nb]))
                found = True
                break
        if not found:
            stack.pop()
    return None


def _decompose_greedy(edges, deadline):
    """Greedy cycle decomposition. Returns list of pieces."""
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    while remaining and time.time() < deadline:
        # Build adjacency
        adj = {}
        for u, v in remaining:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Find vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        # Try to find a cycle, prefer starting from high-degree vertices
        candidates.sort(key=lambda v: len(adj[v]), reverse=True)
        
        best_cycle = None
        for cand in candidates[:10]:
            cycle = _find_cycle_from(adj, cand)
            if cycle and len(cycle) >= 3:
                if best_cycle is None or len(cycle) > len(best_cycle):
                    best_cycle = cycle
                if len(best_cycle) >= 8:  # good enough
                    break
        
        if best_cycle is None:
            break
        
        # Remove cycle edges
        for i in range(len(best_cycle)):
            u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = tuple(sorted((u, v)))
            remaining.discard(e)
        
        pieces.append(best_cycle)
    
    # Remaining edges
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    deadline = time.time() + 140  # safety margin
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Allocate time: more for larger graphs
        graphs_left = len(specs) - i
        time_per = min((deadline - time.time()) / graphs_left, 40)
        
        pieces = _decompose_greedy(edges, time.time() + time_per)
        results.append(pieces)
    
    return results
