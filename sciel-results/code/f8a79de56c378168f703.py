"""Construct 5-regular graph with girth 9 using random regular graph generation and repair."""
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Moore bound for odd girth
    moore = 1
    for i in range(1, (g - 1) // 2 + 1):
        moore += k * (k - 1) ** i
        
    def generate_random_regular(n, degree):
        """Generate a random degree-regular graph using configuration model."""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            attempts += 1
            stubs = []
            for v in range(n):
                for _ in range(degree):
                    stubs.append(v)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None

    def find_short_cycle(n, adj, max_len):
        """
        Find a cycle of length < max_len.
        Returns a list of vertices in the cycle, or None if none found.
        Uses BFS from each vertex.
        """
        # To speed up, we can just check for cycles of length 3, 4, 5...
        # But generic BFS is safer.
        # Optimization: Use visit_id to avoid clearing arrays
        visit_id = 0
        visited = [0] * n
        dist = [0] * n
        parent = [0] * n
        
        for start in range(n):
            visit_id += 1
            visited[start] = visit_id
            dist[start] = 0
            parent[start] = -1
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= max_len - 1:
                    continue
                for v in adj[u]:
                    if visited[v] == 0:
                        visited[v] = visit_id
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        # Cycle length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < max_len:
                            # Reconstruct cycle
                            # Path from u to start
                            path_u = []
                            curr = u
                            while curr != -1:
                                path_u.append(curr)
                                if curr == start:
                                    break
                                curr = parent[curr]
                            
                            # Path from v to start
                            path_v = []
                            curr = v
                            while curr != -1:
                                path_v.append(curr)
                                if curr == start:
                                    break
                                curr = parent[curr]
                                
                            # The cycle is path_u + reversed(path_v)
                            # Note: start is included in both, so we might have duplicates
                            # But we just need *a* cycle to break.
                            # We can return the nodes involved.
                            # For repair, we need the edges.
                            # Let's return the list of nodes in the cycle.
                            cycle_nodes = path_u + path_v[::-1]
                            # Remove duplicates? No, the path logic handles it.
                            # Actually, if u and v are in the same BFS tree,
                            # the path from u to start and v to start meet at start.
                            # The cycle is u -> ... -> start -> ... -> v -> u?
                            # No, the edge (u, v) closes the cycle.
                            # The cycle is path_u (u to start) + edge (start to ... to v)?
                            # Wait, the edge is (u, v).
                            # The path from u to start is P_u.
                            # The path from v to start is P_v.
                            # The cycle is P_u + P_v + (u, v).
                            # Length = len(P_u) + len(P_v) + 1.
                            # dist[u] = len(P_u) - 1.
                            # So cycle_len = dist[u] + dist[v] + 1. Correct.
                            # We return the nodes.
                            return cycle_nodes
        return None

    def try_swap(adj, u1, v1, u2, v2):
        """
        Try to swap edges (u1, v1) and (u2, v2) with (u1, u2) and (v1, v2).
        Returns True if swap is valid and performed.
        """
        # Check if new edges exist
        if u2 in adj[u1] or v2 in adj[v1]:
            return False
        if u1 in adj[u2] or v1 in adj[v2]: # Symmetric check, redundant but safe
             pass
        
        # Check for loops
        if u1 == u2 or v1 == v2:
            return False
            
        # Perform swap
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        adj[u1].append(u2)
        adj[u2].append(u1)
        adj[v1].append(v2)
        adj[v2].append(v1)
        
        return True

    def revert_swap(adj, u1, v1, u2, v2):
        """Revert the swap."""
        adj[u1].remove(u2)
        adj[u2].remove(u1)
        adj[v1].remove(v2)
        adj[v2].remove(v1)
        
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)

    def repair_graph(n, adj, k, g, time_limit):
        """
        Local search to remove short cycles.
        """
        while time.time() < time_limit:
            cycle_nodes = find_short_cycle(n, adj, g)
            if cycle_nodes is None:
                return adj
            
            # Pick two edges in the cycle to swap
            # Cycle nodes: [u, ..., v, ...]
            # We need to identify edges.
            # Let's pick random pairs of edges from the cycle.
            # This is heuristic.
            
            # To make it faster, we can just try to swap edges incident to nodes in the cycle
            # with edges outside the cycle.
            
            # Simple strategy:
            # Pick a random node in the cycle, say u.
            # Pick a neighbor v in the cycle.
            # Pick another node w in the cycle (not u, v).
            # Pick a neighbor x of w (not in cycle).
            # Try swap (u, v) and (w, x).
            
            # This is getting complex. Let's use a simpler global random swap on edges
            # that are part of the cycle.
            
            # Identify edges in the cycle
            cycle_set = set(cycle_nodes)
            cycle_edges = []
            for u in cycle_nodes:
                for v in adj[u]:
                    if v in cycle_set and u < v:
                        cycle_edges.append((u, v))
            
            if not cycle_edges:
                # Should not happen
                return None
                
            # Try a few random swaps involving cycle edges
            success = False
            for _ in range(10):
                # Pick a cycle edge (u1, v1)
                u1, v1 = random.choice(cycle_edges)
                
                # Pick another edge (u2, v2) to swap with
                # It should be disjoint from (u1, v1) to avoid immediate 2-cycles or loops
                # and ideally not in the cycle to break it.
                
                # Pick a random node u2 from the graph
                u2 = random.randint(0, n - 1)
                if u2 == u1 or u2 == v1:
                    continue
                    
                # Pick a neighbor v2 of u2
                if not adj[u2]:
                    continue
                v2 = random.choice(adj[u2])
                
                if v2 == u1 or v2 == v1:
                    continue
                    
                # Check if (u2, v2) is in cycle? If so, swapping two cycle edges might not break it.
                # We want to swap a cycle edge with a non-cycle edge.
                if v2 in cycle_set and u2 in cycle_set:
                    continue
                
                # Attempt swap
                if try_swap(adj, u1, v1, u2, v2):
                    # Check if this created a new short cycle immediately?
                    # To save time, we assume it's an improvement or neutral.
                    # We will verify in the next loop iteration.
                    success = True
                    break
                else:
                    # Swap failed (invalid), do nothing
                    pass
            
            if not success:
                # If we can't find a valid swap, try a different cycle or restart?
                # For now, break and try a new random graph
                return None

        return None

    # Main search loop
    # Start near known minimum
    base_n = 1100
    
    for offset in range(0, 400, 10):
        n = base_n + offset
        if (n * k) % 2 != 0:
            continue
            
        if time.time() > deadline - 20:
            break
            
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
                
            # Repair
            repaired_adj = repair_graph(n, adj, k, g, deadline - 5)
            if repaired_adj:
                # Final verification
                if find_short_cycle(n, repaired_adj, g) is None:
                    edges = []
                    for u in range(n):
                        for v in repaired_adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges

    # Fallback: LUW construction (kept from original)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
