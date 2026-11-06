"""Decompose fixed instance graphs into few cycles + edges (Erdős-Gallai style)."""
import time
from collections import defaultdict, deque
from random import Random

# EVOLVE-BLOCK-START
def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycles(adj, n, max_cycles=5000, time_limit=10.0):
    """Find many edge-disjoint cycles greedily using DFS for simple cycles."""
    remaining = defaultdict(set)
    for u in range(n):
        remaining[u] = set(adj[u])

    cycles = []
    start_time = time.time()

    def dfs_find_cycle(start):
        # Try to find a cycle starting and ending at 'start'
        stack = [(start, [start])]
        visited_in_path = {start}
        # Use a simple DFS but we need to ensure we find a cycle back to start
        # Better: DFS from start, looking for an edge back to start.
        # Actually, standard cycle finding: pick a vertex, DFS, if we hit a visited vertex (in current path), we have a cycle.
        # But we want edge-disjoint, so we remove edges as we go.
        
        # Let's use a recursive DFS to find any cycle.
        # To avoid deep recursion issues, we can use iterative or just rely on small n.
        # n is at most 30, so recursion depth is fine.
        
        def _dfs(u, path, used_edges):
            if time.time() - start_time > time_limit:
                return None
            for v in list(remaining[u]):
                if v == start:
                    if len(path) >= 2:  # Need at least 3 vertices total for a simple cycle
                        return path[:]
                    # If len(path)==1, then u==start, and edge (start,start) is a self-loop, which doesn't exist in simple graphs
                    # If len(path)==2, path=[start, u], edge (u, start) closes a 2-cycle, which is not allowed (need >=3 distinct vertices)
                    # Wait, the problem says "at least three distinct vertices". A 2-cycle is not a simple cycle in simple graphs anyway.
                    # In a simple graph, a cycle must have length >= 3.
                    continue
                if v not in visited_in_path:
                    visited_in_path.add(v)
                    path.append(v)
                    res = _dfs(v, path, used_edges)
                    if res:
                        return res
                    path.pop()
                    visited_in_path.remove(v)
            return None

        return _dfs(start, [start], set())

    # Iterate until no more cycles found or limit reached
    while True:
        if time.time() - start_time > time_limit:
            break
        found = False
        # Try to find a cycle starting from each vertex
        for u in range(n):
            if time.time() - start_time > time_limit:
                break
            if not remaining[u]:
                continue
            cycle = dfs_find_cycle(u)
            if cycle:
                # cycle is a list of vertices [v0, v1, ..., vk] where vk==v0? No, my DFS returns path without closing vertex.
                # The path is [v0, v1, ..., vk] and the edge (vk, v0) exists.
                # So the cycle vertices are cycle list itself.
                # Let's verify: dfs_find_cycle returns path when it finds edge (u, start).
                # path is [start, ..., u]. The edge (u, start) closes it.
                # So the cycle is path.
                cycles.append(cycle)
                # Remove edges from remaining
                for i in range(len(cycle) - 1):
                    u1, u2 = cycle[i], cycle[i+1]
                    remaining[u1].discard(u2)
                    remaining[u2].discard(u1)
                # Remove closing edge
                u1, u2 = cycle[-1], cycle[0]
                remaining[u1].discard(u2)
                remaining[u2].discard(u1)
                found = True
                break
        if not found:
            break
    
    return cycles, remaining


def _decompose(edges, n, time_limit=10.0):
    adj = _build_adj(edges, n)
    cycles, remaining = _find_cycles(adj, n, time_limit=time_limit)
    
    # Remaining edges are single-edge pieces
    pieces = [c for c in cycles]
    for u in range(n):
        for v in remaining[u]:
            if u < v:
                pieces.append([u, v])
    
    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    start_time = time.time()
    results = []
    
    for n, p, seed in specs:
        if time.time() - start_time > 140:
            # Safety margin: return trivial decomposition for remaining graphs
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = _instance(n, p, seed)
        # Allocate remaining time minus buffer for this graph
        remaining_time = 140 - (time.time() - start_time)
        # Use a portion of remaining time for this graph
        alloc_time = min(remaining_time / (len(specs) - len(results)), 10.0)
        if alloc_time < 0.1:
            alloc_time = 0.1
            
        pieces = _decompose(edges, n, time_limit=alloc_time)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
