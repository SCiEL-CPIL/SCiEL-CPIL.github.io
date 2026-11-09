import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, leaving 5s for safety
    
    # Fallback: LUW construction
    def luw_construct(k, g):
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

    # If k or g are not 3 and 16, just use LUW
    if k != 3 or g != 16:
        return luw_construct(k, g)
        
    # Search for smaller graph
    # Moore bound for (3, 16) is 510. Start at 512.
    min_n = 512
    max_n = 1200
    
    # Helper: Generate random 3-regular graph
    def gen_random_3_reg(n):
        # n must be even
        if n % 2 != 0: return None
        # Create 3n half edges
        half_edges = []
        for i in range(n):
            for _ in range(3):
                half_edges.append(i)
        random.shuffle(half_edges)
        
        adj = [[] for _ in range(n)]
        # Pair them up
        edges = set()
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                # Swap to avoid loop
                # Find another edge to swap with
                # Simple retry
                return gen_random_3_reg(n)
            if (v, u) in edges:
                # Multi-edge
                return gen_random_3_reg(n)
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Helper: Check girth
    def has_girth_at_least(adj, g):
        n = len(adj)
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = [start]
            dist[start] = 0
            parent[start] = -1
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But we need to be careful with BFS tree
                        # If v is visited and not parent, it's a back edge
                        # Cycle length = dist[u] + dist[v] + 1
                        # This is correct for undirected graph BFS
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    # Helper: Find shortest cycle
    def find_shortest_cycle(adj, g):
        n = len(adj)
        min_len = g
        cycle = None
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = [start]
            dist[start] = 0
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= min_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        l = dist[u] + dist[v] + 1
                        if l < min_len:
                            min_len = l
                            # Reconstruct cycle?
                            # For now, just return length
                            # We need the cycle to break it
                            # Let's store the edge (u,v)
                            cycle = (u, v)
        return min_len, cycle

    # Main search loop
    for n in range(min_n, max_n, 2):
        if time.time() > deadline - 5:
            break
            
        # Time slice for this n
        slice_end = time.time() + 1.0
        
        found = False
        while time.time() < slice_end and not found:
            adj = gen_random_3_reg(n)
            if not adj: continue
            
            # Check if already good
            if has_girth_at_least(adj, g):
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                found = True
                break
            
            # Local search
            # Try to increase girth
            best_adj = adj
            best_girth = 0
            # Estimate girth
            gl, _ = find_shortest_cycle(adj, g)
            best_girth = gl
            
            for _ in range(1000): # Max swaps per restart
                if time.time() > slice_end: break
                
                # Find short cycle
                gl, (u, v) = find_shortest_cycle(best_adj, g)
                if gl >= g:
                    # Success
                    edges = []
                    for i in range(n):
                        for j in best_adj[i]:
                            if i < j:
                                edges.append((i, j))
                    found = True
                    break
                                
                # Try to break cycle (u, v)
                # Pick a random edge (w, x) not in cycle
                # For simplicity, pick random edge from u's neighbors
                # This is a heuristic
                
                # Get neighbors of u
                nu = best_adj[u]
                # Pick a neighbor w != v
                candidates = [x for x in nu if x != v]
                if not candidates: continue
                w = random.choice(candidates)
                
                # Remove (u, v) and (u, w) ? No, we need to swap edges.
                # Standard 2-switch:
                # Edges (u, v) and (w, x) -> (u, w) and (v, x)
                # We have (u, v). We need another edge (w, x).
                # Let's pick a random edge from the graph.
                # To be efficient, pick a neighbor of v?
                # Let's pick a random edge (w, x) from the adjacency list.
                # We need to ensure (u, w) and (v, x) are not existing edges.
                
                # Pick w from neighbors of v?
                # If we swap (u,v) with (w,x), we need w, x.
                # Let's pick w from neighbors of v (w != u).
                nv = best_adj[v]
                candidates_w = [x for x in nv if x != u]
                if not candidates_w: continue
                w = random.choice(candidates_w)
                
                # Now we have edges (u,v) and (v,w).
                # We want to break the cycle containing (u,v).
                # Swapping (u,v) and (v,w) is not a 2-switch, it's just moving u to w?
                # No, (u,v) and (v,w) share v.
                # We need two disjoint edges.
                
                # Let's pick a random edge (a, b) from the graph.
                # Then swap (u, v) and (a, b) to (u, a) and (v, b).
                # Check validity.
                
                # Random edge
                a = random.randint(0, n-1)
                if not best_adj[a]: continue
                b = random.choice(best_adj[a])
                
                # Edges: (u, v) and (a, b)
                # New: (u, a) and (v, b)
                # Check if (u, a) exists
                if a in best_adj[u]: continue
                if b in best_adj[v]: continue
                # Check if (u, a) is a loop
                if u == a: continue
                if v == b: continue
                
                # Apply swap
                # Remove (u, v)
                best_adj[u].remove(v)
                best_adj[v].remove(u)
                # Remove (a, b)
                best_adj[a].remove(b)
                best_adj[b].remove(a)
                
                # Add (u, a)
                best_adj[u].append(a)
                best_adj[a].append(u)
                # Add (v, b)
                best_adj[v].append(b)
                best_adj[b].append(v)
                
                # Check girth
                new_gl, _ = find_shortest_cycle(best_adj, g)
                if new_gl > best_girth:
                    best_girth = new_gl
                else:
                    # Revert
                    best_adj[u].remove(a)
                    best_adj[a].remove(u)
                    best_adj[v].remove(b)
                    best_adj[b].remove(v)
                    
                    best_adj[u].append(v)
                    best_adj[v].append(u)
                    best_adj[a].append(b)
                    best_adj[b].append(a)

        if found:
            return n, edges

    # Fallback
    return luw_construct(k, g)
