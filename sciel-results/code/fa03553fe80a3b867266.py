import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback to seed if we fail
    def seed_construct(k, g):
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

    def count_short_cycles(adj, n, max_len=4):
        """Count cycles of length <= max_len. Returns (count, list_of_cycles)"""
        count = 0
        cycles = []
        # Triangles
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            count += 1
                            cycles.append([i, j, k])
        # 4-cycles
        # i - j - k - l - i
        # To avoid duplicates, enforce i < j < k < l? No, just i < j and k, l distinct.
        # Efficient way: for each edge (i, j), look for common neighbors of i and j? No, that's triangles.
        # For 4-cycles: i-j-k-l-i.
        # Count pairs of neighbors (j, l) of i such that j and l are connected via a common neighbor k?
        # No, j-k and l-k.
        # So for each i, for each pair of neighbors (j, l), check if they share a common neighbor k (k != i).
        for i in range(n):
            neighbors_i = adj[i]
            for a_idx in range(len(neighbors_i)):
                for b_idx in range(a_idx + 1, len(neighbors_i)):
                    j = neighbors_i[a_idx]
                    l = neighbors_i[b_idx]
                    # Common neighbors of j and l, excluding i
                    common = set(adj[j]) & set(adj[l])
                    common.discard(i)
                    if common:
                        # Each common neighbor k forms a 4-cycle i-j-k-l-i
                        # But we might count each 4-cycle multiple times.
                        # A 4-cycle i-j-k-l-i is counted for:
                        # i: neighbors j, l. common k.
                        # j: neighbors i, k. common l.
                        # k: neighbors j, l. common i.
                        # l: neighbors k, i. common j.
                        # So each 4-cycle is counted 4 times.
                        count += len(common)
                        for k in common:
                            cycles.append([i, j, k, l])
        return count // 4 if max_len >= 4 else count, cycles # Adjusting for 4-cycle overcounting is tricky, let's just return raw count for heuristic
    
    def find_one_short_cycle(adj, n, g):
        """Find one cycle of length < g. Returns list of vertices or None."""
        # Check for triangles first
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return [i, j, k]
        # Check for 4-cycles
        for i in range(n):
            neighbors_i = adj[i]
            for a_idx in range(len(neighbors_i)):
                for b_idx in range(a_idx + 1, len(neighbors_i)):
                    j = neighbors_i[a_idx]
                    l = neighbors_i[b_idx]
                    common = set(adj[j]) & set(adj[l])
                    common.discard(i)
                    if common:
                        k = next(iter(common))
                        return [i, j, k, l]
        # Check for longer cycles (5 to g-1) using BFS from each node
        # This is expensive. Only do it if no short cycles found.
        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            queue = [start]
            dist[start] = 0
            while queue:
                u = queue.pop(0)
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u] and v != start:
                        length = dist[u] + dist[v] + 1
                        if length < g:
                            # Reconstruct
                            path1 = []
                            curr = u
                            while curr != -1:
                                path1.append(curr)
                                if curr == start: break
                                curr = prev[curr]
                            path1.reverse()
                            path2 = []
                            curr = v
                            while curr != -1:
                                path2.append(curr)
                                if curr == start: break
                                curr = prev[curr]
                            return path1 + path2[1:]
                    elif v == start and dist[u] + 1 < g:
                        length = dist[u] + 1
                        if length < g:
                            path = []
                            curr = u
                            while curr != -1:
                                path.append(curr)
                                if curr == start: break
                                curr = prev[curr]
                            path.reverse()
                            return path
        return None

    def apply_switch(adj, u, v, x, y):
        """Swap (u,v) and (x,y) with (u,x) and (v,y) or (u,y) and (v,x)."""
        # Remove
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        # Try 1
        if x not in adj[u] and y not in adj[v] and u != x and v != y:
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True
        
        # Revert 1
        if x in adj[u]: adj[u].remove(x)
        if u in adj[x]: adj[x].remove(u)
        if y in adj[v]: adj[v].remove(y)
        if v in adj[y]: adj[y].remove(v)
        
        # Try 2
        if y not in adj[u] and x not in adj[v] and u != y and v != x:
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            return True
            
        # Revert 2
        if y in adj[u]: adj[u].remove(y)
        if u in adj[y]: adj[y].remove(u)
        if x in adj[v]: adj[v].remove(x)
        if v in adj[x]: adj[x].remove(v)
        
        # Re-add original
        adj[u].append(v)
        adj[v].append(u)
        adj[x].append(y)
        adj[y].append(x)
        return False

    def generate_random_cubic_graph(n):
        if n % 2 != 0 or n < 4:
            return None
        adj = [[(i-1)%n, (i+1)%n] for i in range(n)]
        for _ in range(1000):
            slots = list(range(n))
            random.shuffle(slots)
            adj_temp = [list(neighs) for neighs in adj]
            success = True
            for i in range(0, n, 2):
                u = slots[i]
                v = slots[i+1]
                if u == v or v in adj_temp[u]:
                    success = False
                    break
                adj_temp[u].append(v)
                adj_temp[v].append(u)
            if success:
                return adj_temp
        return None

    # Main Search
    best_n = None
    best_edges = None
    
    # Try n from 512 upwards
    # We will spend ~5s per n value
    for n in range(512, 1200, 2):
        if time.time() > deadline - 5:
            break
        
        adj = generate_random_cubic_graph(n)
        if not adj:
            continue
            
        n_start = time.time()
        while time.time() - n_start < 5.0:
            if time.time() > deadline - 2:
                break
                
            # Check if valid
            # For speed, check triangles and 4-cycles first
            # If none, do full girth check
            # But full girth check is slow.
            # Let's just try to find a short cycle.
            cycle = find_one_short_cycle(adj, n, g)
            
            if cycle is None:
                # Valid!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            else:
                # Pick an edge in cycle
                # Cycle is [v0, v1, ...]
                # Edges are (v0,v1), etc.
                u = cycle[0]
                v = cycle[1]
                
                # Find a partner edge
                # Pick random edge
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
                
                if not all_edges:
                    break
                    
                x, y = random.choice(all_edges)
                if x == u or x == v or y == u or y == v:
                    continue
                    
                if apply_switch(adj, u, v, x, y):
                    pass
                else:
                    pass

    # If we didn't find a better graph, return seed
    return seed_construct(k, g)
