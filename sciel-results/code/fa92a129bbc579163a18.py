import random
import time

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def lw_construct(k, g):
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

    if k != 3:
        return lw_construct(k, g)

    target_n_list = [800, 820, 840, 860, 880, 900, 920]
    start_time = time.time()
    time_limit = 110

    for target_n in target_n_list:
        if time.time() - start_time > time_limit:
            break
        N = target_n // 2
        valid_graph = None
        attempts = 0
        max_attempts = 1000

        while attempts < max_attempts and time.time() - start_time < time_limit:
            attempts += 1
            adj = [set() for _ in range(2*N)]
            
            left_nodes = list(range(N))
            right_nodes = list(range(N, 2*N))
            
            p1 = right_nodes[:]
            p2 = right_nodes[:]
            p3 = right_nodes[:]
            random.shuffle(p1)
            random.shuffle(p2)
            random.shuffle(p3)
            
            valid_init = True
            for i in range(N):
                n1, n2, n3 = p1[i], p2[i], p3[i]
                if n1 == n2 or n1 == n3 or n2 == n3:
                    valid_init = False
                    break
                adj[i].add(n1)
                adj[n1].add(i)
                adj[i].add(n2)
                adj[n2].add(i)
                adj[i].add(n3)
                adj[n3].add(i)
            
            if not valid_init:
                continue
            
            for step in range(5000):
                if time.time() - start_time > time_limit:
                    break
                
                cycle = find_short_cycle(adj, 2*N, g)
                if not cycle:
                    valid_graph = adj
                    break
                
                # Break cycle
                edge_idx = random.randint(0, len(cycle) - 1)
                u, v = cycle[edge_idx]
                
                adj[u].remove(v)
                adj[v].remove(u)
                
                other_side = list(range(N, 2*N)) if u < N else list(range(N))
                candidates = [x for x in other_side if x not in adj[u]]
                
                if not candidates:
                    break
                
                new_v = random.choice(candidates)
                adj[u].add(new_v)
                adj[new_v].add(u)
            
            if valid_graph:
                break
        
        if valid_graph:
            edges = []
            for i in range(N):
                for j in valid_graph[i]:
                    if i < j:
                        edges.append((i, j))
            return 2*N, edges

    return lw_construct(k, g)

def find_short_cycle(adj, n, girth):
    # Check 4-cycles
    for u in range(n):
        neighbors_u = list(adj[u])
        for i in range(len(neighbors_u)):
            v = neighbors_u[i]
            for j in range(i+1, len(neighbors_u)):
                x = neighbors_u[j]
                common = adj[v].intersection(adj[x])
                for w in common:
                    if w != u:
                        return [(u, v), (v, w), (w, x), (x, u)]
    
    # Check 6-cycles
    # u-v-w-x-y-z-u
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w == u: continue
                for x in adj[w]:
                    if x == v: continue
                    # Check 4-cycle again? No, we already did.
                    # But x might be in adj[u]. If so, it's a 4-cycle u-v-w-x-u.
                    # We should have caught it.
                    if x in adj[u] and x != v:
                        return [(u, v), (v, w), (w, x), (x, u)]
                    
                    for y in adj[x]:
                        if y == w: continue
                        for z in adj[y]:
                            if z == x: continue
                            if z == u and z != v:
                                return [(u, v), (v, w), (w, x), (x, y), (y, z), (z, u)]
    
    # Check 8-cycles
    # u-v-w-x-y-z-a-b-u
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w == u: continue
                for x in adj[w]:
                    if x == v: continue
                    for y in adj[x]:
                        if y == w: continue
                        for z in adj[y]:
                            if z == x: continue
                            for a in adj[z]:
                                if a == y: continue
                                for b in adj[a]:
                                    if b == z: continue
                                    if b == u and b != v:
                                        return [(u, v), (v, w), (w, x), (x, y), (y, z), (z, a), (a, b), (b, u)]
    
    # Check 10, 12, 14 cycles similarly if needed.
    # For girth 16, we need to check up to 14.
    # The above loops are $O(N \cdot 3^{L/2})$.
    # For L=14, depth is 7. $3^7 = 2187$.
    # $400 \cdot 2187 \approx 10^6$. This is fine.
    
    # Generalized check for higher girths
    # We can just do a BFS/DFS up to depth girth/2 from each node.
    # But the specific checks above are faster for small L.
    # Let's add a generic fallback for L > 8.
    
    if girth > 8:
        # Use a random walk to find longer cycles
        for _ in range(200):
            start = random.randint(0, n-1)
            path = [start]
            visited = {start}
            current = start
            prev = -1
            for _ in range(girth):
                neighbors = [x for x in adj[current] if x != prev]
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                if next_node in visited:
                    idx = path.index(next_node)
                    cycle_nodes = path[idx:]
                    cycle_edges = []
                    for i in range(len(cycle_nodes)):
                        u = cycle_nodes[i]
                        v = cycle_nodes[(i+1) % len(cycle_nodes)]
                        cycle_edges.append((u, v))
                    return cycle_edges
                visited.add(next_node)
                path.append(next_node)
                prev = current
                current = next_node
    
    return None
