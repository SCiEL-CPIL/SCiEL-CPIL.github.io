import sys
import time
from collections import deque, defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_bridges(n, adj):
        # adj: list of sets of neighbors
        visited = [False] * n
        disc = [0] * n
        low = [0] * n
        parent = [-1] * n
        bridges = set()
        timer = 0

        def dfs(u):
            nonlocal timer
            visited[u] = True
            disc[u] = timer
            low[u] = timer
            timer += 1
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.add((min(u, v), max(u, v)))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if not visited[i]:
                dfs(i)
        return bridges

    def get_cycle_from_edge(n, adj, e):
        # e is (u, v)
        # Find shortest path from u to v in graph without edge e
        # Using BFS
        u, v = e
        # Temporarily remove e from adj for BFS? 
        # Easier: just skip this edge in BFS neighbor iteration
        visited = {u: None} # node: parent
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if curr == v:
                # Reconstruct path
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = visited[node]
                path.reverse()
                return path
            for neighbor in adj[curr]:
                # Skip the edge we are excluding
                if (curr, neighbor) == e or (neighbor, curr) == e:
                    continue
                if neighbor not in visited:
                    visited[neighbor] = curr
                    queue.append(neighbor)
        return None

    def decompose(n, edges):
        # Build adjacency list
        # adj[u] is a set of neighbors
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # We need to work on a mutable set of edges
        current_edges = set(edges)
        pieces = []
        
        while current_edges:
            if time.time() > deadline:
                break # Timeout safeguard
            
            # 1. Find bridges
            # Rebuild adj for current_edges for bridge finding
            # Note: find_bridges needs n and adj
            # We can just pass the current adj built from current_edges
            curr_adj = defaultdict(set)
            for u, v in current_edges:
                curr_adj[u].add(v)
                curr_adj[v].add(u)
                
            bridges = find_bridges(n, curr_adj)
            
            if bridges:
                # Remove all bridges
                for b in bridges:
                    pieces.append([b[0], b[1]])
                    current_edges.remove(b)
                # Loop back to re-evaluate bridges on the reduced graph
                continue
            
            # 2. No bridges, so graph is 2-edge-connected (or single edge? No, single edge is a bridge)
            # So if current_edges is not empty, it has a cycle.
            
            # Find the cycle with maximum length
            best_cycle = None
            best_len = -1
            
            # Iterate over all edges to find the one that yields the longest cycle
            # This is O(E * (V+E))
            for e in list(current_edges):
                # Optimization: if current best_len is already large, maybe skip? 
                # But we want max, so we must check all unless we find a Hamiltonian cycle?
                # For small graphs, just check all.
                cycle = get_cycle_from_edge(n, curr_adj, e)
                if cycle:
                    # Cycle length is number of edges in the path + 1 (the edge e)
                    # Path is vertices, so edges = len(path) - 1
                    # Total cycle edges = len(path)
                    l = len(cycle)
                    if l > best_len:
                        best_len = l
                        best_cycle = cycle
            
            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges of this cycle from current_edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    if edge in current_edges:
                        current_edges.remove(edge)
            else:
                # Should not happen if no bridges and edges exist
                # Fallback: take any edge as single piece to avoid infinite loop
                if current_edges:
                    e = next(iter(current_edges))
                    pieces.append([e[0], e[1]])
                    current_edges.remove(e)
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        res = decompose(n, edges)
        results.append(res)
        
    return results
