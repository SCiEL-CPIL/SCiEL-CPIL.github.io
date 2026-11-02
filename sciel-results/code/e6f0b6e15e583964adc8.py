import time
import random
import sys
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, (g + 2) // 2)
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

    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))
    
    deadline = time.time() + 110
    
    if k == 4 and g == 10:
        # Try to find the 320 cage
        for target_n in [320, 322, 324, 326, 328, 330]:
            if time.time() > deadline - 20:
                break
            result = try_find_cage(target_n, k, g, deadline - time.time() - 5)
            if result:
                return result
        
        return luw_construct(k, g)
    
    return luw_construct(k, g)

def try_find_cage(n, k, g, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def verify_girth_fast(n, adj, g):
        """BFS from each vertex up to depth g//2. If we find a back-edge creating
        a cycle of length < g, return False. This is O(n * k^(g/2)) which is much
        faster than non-backtracking walks O(n * k^(g-1))."""
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a back-edge; check if it creates a short cycle
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def count_short_cycles(n, adj, g):
        """Count the number of distinct short cycles (length < g) using BFS.
        This serves as a smooth surrogate objective for local search."""
        half_g = g // 2
        cycle_count = 0
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Only count each cycle once: require start < min other vertex
                                # Simple heuristic: count if start is the smallest vertex in the cycle
                                # For speed, just count all and divide by approximation
                                cycle_count += 1
        return cycle_count

    def find_short_cycle_edges(n, adj, g, start=None):
        """Find one short cycle and return its edge set. Returns list of (u,v) edges or None."""
        half_g = g // 2
        starts = range(n) if start is None else [start]
        for s in starts:
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                path1 = []
                                cur = u
                                while cur != s:
                                    path1.append(cur)
                                    cur = parent[cur]
                                path1.append(s)
                                path1.reverse()
                                
                                path2 = []
                                cur = v
                                while cur != s:
                                    path2.append(cur)
                                    cur = parent[cur]
                                path2.append(s)
                                path2.reverse()
                                
                                i = 0
                                while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
                                    i += 1
                                if i == 0:
                                    continue
                                cycle = path1[i:] + path2[i-1::-1]
                                if len(cycle) >= 3:
                                    # Return edges of this cycle
                                    edges = []
                                    for j in range(len(cycle)):
                                        a, b = cycle[j], cycle[(j+1) % len(cycle)]
                                        edges.append((min(a,b), max(a,b)))
                                    return edges
        return None

    def fast_regular_graph(n, k):
        """Generate a random k-regular graph using configuration model."""
        for attempt in range(20000):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None

    def local_search(edges, adj, n, k, g, edge_set, deadline):
        """Local search using 2-opt edge swaps to minimize short cycle count."""
        best_cycle_count = count_short_cycles(n, adj, g)
        if best_cycle_count == 0:
            return n, edges, True
        
        current_cycle_count = best_cycle_count
        stall_count = 0
        no_improve = 0
        temperature = 1.0
        
        while time.time() < deadline - 2:
            # Find a short cycle to break
            cycle_edges = find_short_cycle_edges(n, adj, g)
            if cycle_edges is None:
                # Verify fully
                if verify_girth_fast(n, adj, g):
                    return n, edges, True
                # BFS might have missed it; try again
                continue
            
            # Pick an edge from the cycle to swap
            improved = False
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Pick a random edge from the cycle
                e1 = random.choice(cycle_edges)
                u, v = e1
                
                # Find another edge to swap with
                # Try multiple random edges
                for _ in range(20):
                    if time.time() > deadline - 2:
                        break
                    
                    # Pick a random edge not incident to u or v
                    e2 = random.choice(list(edge_set))
                    a, b = e2
                    if a in (u, v) or b in (u, v):
                        continue
                    if e2 == e1:
                        continue
                    
                    # Check if new edges would be duplicates or loops
                    new_e1 = (min(u, a), max(u, a))
                    new_e2 = (min(v, b), max(v, b))
                    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                        continue
                    if new_e1 in edge_set or new_e2 in edge_set:
                        continue
                    if new_e1 == new_e2:
                        continue
                    
                    # Perform the swap
                    old_e1 = (min(u, v), max(u, v))
                    old_e2 = (min(a, b), max(a, b))
                    new_edge_set = set(edge_set)
                    new_edge_set.discard(old_e1)
                    new_edge_set.discard(old_e2)
                    new_edge_set.add(new_e1)
                    new_edge_set.add(new_e2)
                    
                    # Build new adjacency
                    new_adj = [[] for _ in range(n)]
                    for x, y in new_edge_set:
                        new_adj[x].append(y)
                        new_adj[y].append(x)
                    
                    # Quick check: count short cycles (approximate by checking a few vertices)
                    # For speed, do a partial check
                    partial_count = 0
                    check_vertices = random.sample(range(n), min(n, 40))
                    half_g = g // 2
                    for s in check_vertices:
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[s] = 0
                        queue = [s]
                        head = 0
                        while head < len(queue):
                            uu = queue[head]
                            head += 1
                            if dist[uu] >= half_g:
                                continue
                            for vv in new_adj[uu]:
                                if dist[vv] == -1:
                                    dist[vv] = dist[uu] + 1
                                    parent[vv] = uu
                                    queue.append(vv)
                                else:
                                    if vv != parent[uu] and uu != parent[vv]:
                                        cycle_len = dist[uu] + dist[vv] + 1
                                        if cycle_len < g:
                                            partial_count += 1
                                            break
                    
                    # Accept if it looks better (with some probability for exploration)
                    if partial_count < current_cycle_count / max(1, n // 40):
                        # Accept the swap
                        edge_set = new_edge_set
                        adj = new_adj
                        edges = [tuple(e) for e in edge_set]
                        current_cycle_count = partial_count * (n // 40)  # Scale up estimate
                        best_cycle_count = min(best_cycle_count, current_cycle_count)
                        improved = True
                        no_improve = 0
                        break
                    else:
                        # Occasionally accept worse moves (simulated annealing)
                        if random.random() < 0.1 * temperature:
                            edge_set = new_edge_set
                            adj = new_adj
                            edges = [tuple(e) for e in edge_set]
                            current_cycle_count = partial_count * (n // 40)
                            no_improve += 1
                            break
            
            if improved:
                stall_count = 0
            else:
                stall_count += 1
                no_improve += 1
            
            # Cool down
            temperature = max(0.05, temperature * 0.95)
            
            if no_improve > 50:
                # Restart with new random graph
                new_edges = fast_regular_graph(n, k)
                if new_edges is None:
                    break
                adj = build_adj(new_edges, n)
                edge_set = set((min(u,v), max(u,v)) for u,v in new_edges)
                edges = new_edges
                current_cycle_count = count_short_cycles(n, adj, g)
                best_cycle_count = min(best_cycle_count, current_cycle_count)
                no_improve = 0
                stall_count = 0
                temperature = 1.0
            
            if stall_count > 100:
                break
        
        # Final verification
        if verify_girth_fast(n, adj, g):
            return n, edges, True
        
        return None

    # Generate initial random regular graph
    edges = fast_regular_graph(n, k)
    if edges is None:
        return None
    
    adj = build_adj(edges, n)
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    result = local_search(edges, adj, n, k, g, edge_set, deadline)
    return result
