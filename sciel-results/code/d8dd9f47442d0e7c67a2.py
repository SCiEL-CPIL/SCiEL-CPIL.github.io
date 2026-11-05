"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    q = k
    # Use smallest d to get a compact graph, then repair short cycles
    d = 3  # LUW CD(3,q) has girth ~8 for q=4
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (0,) * (d + 1))
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    if n == 0:
        return 0, []
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify 4-regular
    if any(len(adj[i]) != k for i in range(n)):
        return n, edges
    
    # If girth already >= g, return immediately
    def check_girth(adj, n, min_girth):
        """Check if graph has no cycle shorter than min_girth using BFS from each vertex."""
        for src in range(n):
            # BFS with parent tracking to avoid backtracking
            visited = {src: (None, 0)}  # node -> (parent, depth)
            queue = deque([(src, -1, 0)])
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= min_girth - 1:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        # Found a cycle of length depth + 1
                        if depth + 1 < min_girth:
                            return False
                    else:
                        visited[nb] = (node, depth + 1)
                        queue.append((nb, node, depth + 1))
        return True
    
    if check_girth(adj, n, g):
        return n, edges
    
    # Repair: find and break short cycles
    # Strategy: iteratively find a short cycle, remove an edge from it,
    # then reconnect the two endpoints to other vertices to maintain regularity
    # without creating new short cycles.
    
    def find_short_cycle(adj, n, min_girth):
        """Find a cycle of length < min_girth. Returns list of nodes or None."""
        for src in range(n):
            visited = {src: (None, 0)}
            queue = deque([(src, -1, 0)])
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= min_girth - 1:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        if depth + 1 < min_girth:
                            # Reconstruct cycle
                            cycle = [node]
                            cur = node
                            while cur != src:
                                cur, _ = visited[cur]
                                cycle.append(cur)
                            return cycle
                    else:
                        visited[nb] = (node, depth + 1)
                        queue.append((nb, node, depth + 1))
        return None
    
    def would_create_short_cycle(adj, n, u, v, min_girth):
        """Check if adding edge (u,v) would create a cycle shorter than min_girth."""
        # BFS from u to v avoiding the edge (u,v) - if path length < min_girth - 1, adding creates short cycle
        if u in adj[v]:
            return True  # already connected
        # Check if there's a path from u to v of length < min_girth - 1
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= min_girth - 2:
                continue
            for nb in adj[node]:
                if nb == v:
                    return True  # path of length depth+1 < min_girth-1
                if nb not in visited:
                    visited[nb] = depth + 1
                    queue.append((nb, depth + 1))
        return False
    
    def try_repair(adj, n, k, g, deadline):
        """Attempt to repair short cycles. Returns (success, new_edges)."""
        edges_set = set()
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges_set.add((i, j))
        
        for iteration in range(500):
            if time.time() > deadline:
                return False, None
            
            cycle = find_short_cycle(adj, n, g)
            if cycle is None:
                # No more short cycles
                # Verify
                if check_girth(adj, n, g):
                    # Rebuild edges
                    new_edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                new_edges.append((i, j))
                    return True, new_edges
                else:
                    return False, None
            
            # Pick an edge from the cycle to remove
            # Try each edge in the cycle
            removed = False
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if (u, v) not in edges_set and (v, u) not in edges_set:
                    continue
                
                # Remove edge (u, v)
                adj[u].remove(v)
                adj[v].remove(u)
                if (u, v) in edges_set:
                    edges_set.remove((u, v))
                else:
                    edges_set.remove((v, u))
                
                # Now u and v have degree k-1. Need to add edges to restore degree k.
                # Find candidates for u and v
                u_candidates = []
                v_candidates = []
                for cand in range(n):
                    if cand == u or cand == v:
                        continue
                    if cand in adj[u]:
                        continue
                    if would_create_short_cycle(adj, n, u, cand, g):
                        continue
                    u_candidates.append(cand)
                
                for cand in range(n):
                    if cand == u or cand == v:
                        continue
                    if cand in adj[v]:
                        continue
                    if would_create_short_cycle(adj, n, v, cand, g):
                        continue
                    v_candidates.append(cand)
                
                # Need to add one edge to u and one edge to v
                # They can be the same vertex or different
                found = False
                for uc in u_candidates:
                    for vc in v_candidates:
                        if uc == vc:
                            # Add edge u-uc and v-vc (same vertex)
                            adj[u].append(uc)
                            adj[vc].append(u)
                            adj[v].append(uc)
                            adj[vc].append(v)
                            edges_set.add((min(u, uc), max(u, uc)))
                            edges_set.add((min(v, uc), max(v, uc)))
                            found = True
                            break
                        elif uc not in adj[v] and vc not in adj[u]:
                            # Add edge u-uc and v-vc
                            if uc == v or vc == u:
                                continue
                            adj[u].append(uc)
                            adj[uc].append(u)
                            adj[v].append(vc)
                            adj[vc].append(v)
                            edges_set.add((min(u, uc), max(u, uc)))
                            edges_set.add((min(v, vc), max(v, vc)))
                            found = True
                            break
                    if found:
                        break
                if found:
                    removed = True
                    break
                
                if not found:
                    # Restore edge
                    adj[u].append(v)
                    adj[v].append(u)
                    edges_set.add((min(u, v), max(u, v)))
            
            if not removed:
                # Couldn't fix any edge in this cycle
                return False, None
        
        return False, None
    
    success, new_edges = try_repair(adj, n, k, g, deadline)
    if success and new_edges is not None:
        return n, new_edges
    
    # Fallback: return original if it's at least valid (4-regular)
    return n, edges
# EVOLVE-BLOCK-END
