import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110  # 10s safety margin

    # General fallback: LUW construction
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

    # For the hard case k=3, g=16, use a structured approach
    if k == 3 and g == 16:
        # Strategy: Random voltage lift of Petersen graph + 2-switch local search
        # This explores a structured space rather than pure random 3-regular graphs.
        
        random.seed(42)
        
        # Petersen graph edges (standard labeling)
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        n_base = 10
        m_base = 15
        
        def build_lift(n_lift):
            """
            Build a 3-regular graph by lifting the Petersen graph with voltage n_lift.
            Vertices are (base_vertex, layer).
            Edges connect (u, l) to (v, (l + voltage) % n_lift).
            """
            edges = []
            # Assign random voltages to each base edge
            # For undirected edges, we just pick a voltage.
            # The lift is well-defined if we orient edges consistently, but for 3-regular
            # we can just assign a voltage to each edge.
            
            # We need to ensure the lift is simple (no loops or multi-edges).
            # A loop occurs if u==v and voltage==0. In Petersen, u!=v for all edges.
            # Multi-edges occur if two different base edges (u,v) and (u',v') 
            # result in the same lifted edge.
            # Since the base is simple, and we lift, multi-edges can only happen
            # if the graph is not "voltage-complete" in a way that causes collisions.
            # For a tree-like base (Petersen is not a tree), collisions are possible
            # but rare for large n_lift. We will check for simplicity.
            
            voltages = [random.randint(0, n_lift - 1) for _ in range(m_base)]
            
            # Map base edge index to voltage
            # For each base edge (u, v), the lifted edges are:
            # (u, l) - (v, (l + volt[i]) % n_lift) for l in 0..n_lift-1
            # Note: Since the edge is undirected, we treat it as one connection.
            
            for i, (u, v) in enumerate(base_edges):
                vol = voltages[i]
                for l in range(n_lift):
                    u_lift = u * n_lift + l
                    v_lift = v * n_lift + (l + vol) % n_lift
                    edges.append((u_lift, v_lift))
            
            n_total = n_base * n_lift
            return n_total, edges

        def is_simple_and_regular(edges, n):
            """Check if graph is simple and 3-regular."""
            if len(edges) != n * 3 // 2:
                return False
            
            adj = [set() for _ in range(n)]
            for u, v in edges:
                if u == v:
                    return False
                if v in adj[u]:
                    return False
                adj[u].add(v)
                adj[v].add(u)
            
            for i in range(n):
                if len(adj[i]) != 3:
                    return False
            return True

        def check_girth_fast(adj, n, required_girth):
            """
            Check if girth is at least required_girth.
            Returns True if girth >= required_girth, else False.
            Uses BFS from each vertex up to depth required_girth // 2.
            """
            max_depth = required_girth // 2
            # Optimization: If we find a cycle shorter than required_girth, return False.
            # We don't need to find the exact girth, just verify lower bound.
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < required_girth:
                                return False
            return True

        def find_short_cycle(adj, n, required_girth):
            """
            Find a cycle shorter than required_girth.
            Returns (cycle_len, cycle_nodes) or (inf, None).
            """
            best_len = float('inf')
            best_cycle = None
            max_depth = required_girth // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < best_len:
                                best_len = cycle_len
                                # Reconstruct cycle
                                # Path from start to u
                                path_u = []
                                cur = u
                                while cur != -1:
                                    path_u.append(cur)
                                    cur = parent[cur]
                                path_u.reverse()
                                
                                # Path from start to v
                                path_v = []
                                cur = v
                                while cur != -1:
                                    path_v.append(cur)
                                    cur = parent[cur]
                                path_v.reverse()
                                
                                # Cycle is path_u + reversed path_v (excluding start)
                                # Actually, the cycle is u -> ... -> start -> ... -> v -> u
                                # Wait, the edge is (u, v).
                                # The cycle is: u, parent[u], ..., start, ..., parent[v], v
                                # Let's just return the length for now, and maybe the edges.
                                # For 2-switch, we need the edges of the cycle.
                                # Let's store the nodes.
                                cycle_nodes = path_u + path_v[::-1]
                                # The cycle is formed by the path from u to start, 
                                # then path from start to v, then edge v-u.
                                # Actually, path_u is start->...->u. path_v is start->...->v.
                                # The cycle is u -> ... -> start -> ... -> v -> u.
                                # Nodes: path_u (reversed to go u->start) + path_v (start->v)
                                # But we already have path_u as start->u.
                                # So cycle: path_u[::-1] + path_v[1:]  (excluding start from path_v)
                                # Let's verify: 
                                # path_u = [s, ..., u]
                                # path_v = [s, ..., v]
                                # Cycle: u, ..., s, ..., v, u
                                # Nodes: path_u[::-1] + path_v[1:]
                                best_cycle = path_u[::-1] + path_v[1:]
            return best_len, best_cycle

        def get_cycle_edges(cycle_nodes, adj):
            """
            Given a list of nodes forming a cycle (u0, u1, ..., uk-1, u0),
            return the list of edges in the cycle.
            """
            edges = []
            k = len(cycle_nodes)
            for i in range(k):
                u = cycle_nodes[i]
                v = cycle_nodes[(i + 1) % k]
                # Ensure u < v for consistency? No, just return as is.
                # The graph is undirected.
                edges.append((u, v))
            return edges

        def apply_2_switch(edges, edge1, edge2):
            """
            Apply a 2-switch: remove edges (a,b) and (c,d), add (a,c) and (b,d).
            Assumes a!=b, c!=d, and all distinct.
            """
            a, b = edge1
            c, d = edge2
            new_edges = []
            for e in edges:
                if e == (a, b) or e == (b, a):
                    continue
                if e == (c, d) or e == (d, c):
                    continue
                new_edges.append(e)
            new_edges.append((a, c))
            new_edges.append((b, d))
            return new_edges

        def is_valid_2_switch(adj, edge1, edge2):
            """
            Check if a 2-switch results in a simple graph.
            edge1 = (a,b), edge2 = (c,d).
            New edges: (a,c) and (b,d).
            Must not be loops: a!=c, b!=d.
            Must not be existing edges: (a,c) not in adj, (b,d) not in adj.
            Must not be the edges being removed (handled by distinctness).
            """
            a, b = edge1
            c, d = edge2
            if a == c or b == d:
                return False
            if a == d or b == c:
                return False
            if c in adj[a]:
                return False
            if d in adj[b]:
                return False
            return True

        # Main loop
        best_n = float('inf')
        best_edges = None
        
        # Start with a reasonable lift size.
        # The Petersen graph has girth 5.
        # A random lift of size L has expected girth roughly 2*log_{2}(L) + constant?
        # Actually, for a fixed base graph, the girth of the lift grows with L.
        # For L=100, n=1000. 
        # Let's try L=100 (n=1000) first.
        
        # We want to find a graph with n close to 936.
        # Let's try L=94 (n=940), L=96 (n=960), etc.
        
        # Actually, the lift size must be such that n is even?
        # n = 10 * L. Always even.
        
        # Let's try to find ANY valid graph first, then minimize n.
        # But we have a time limit.
        
        # Strategy:
        # 1. Try L=94, 96, 98, 100...
        # 2. For each L, generate random lifts.
        # 3. Use 2-switch local search to improve girth.
        
        for L in range(94, 120, 2):
            n_total = 10 * L
            if time.time() > deadline - 15:
                break
            
            found = False
            for attempt in range(50):
                if time.time() > deadline - 15:
                    break
                
                n, edges = build_lift(L)
                if not is_simple_and_regular(edges, n):
                    continue
                
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Check girth
                if check_girth_fast(adj, n, g):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                
                # If girth is close, try to improve with 2-switches
                # Find shortest cycle
                short_len, cycle_nodes = find_short_cycle(adj, n, g)
                
                if short_len < g:
                    # Try to break the cycle using 2-switches
                    # This is complex. A simpler heuristic:
                    # Pick a random edge from the cycle, and a random edge elsewhere.
                    # Perform a 2-switch if valid.
                    # Repeat for a few steps.
                    
                    cycle_edges = get_cycle_edges(cycle_nodes, adj)
                    
                    improved = False
                    for _ in range(20): # Max 20 switches per attempt
                        if time.time() > deadline - 15:
                            break
                        
                        # Pick a random edge from the cycle
                        e1 = random.choice(cycle_edges)
                        # Pick a random edge from the whole graph
                        e2 = random.choice(edges)
                        
                        # Ensure e1 != e2
                        if (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0]):
                            continue
                        
                        if is_valid_2_switch(adj, e1, e2):
                            new_edges = apply_2_switch(edges, e1, e2)
                            
                            # Check if new graph is simple (2-switch guarantees no multi-edges if valid)
                            # But we should verify regularity (it is preserved) and simplicity.
                            # is_valid_2_switch checks for new edge conflicts.
                            
                            # Update adj
                            a, b = e1
                            c, d = e2
                            adj[a].remove(b)
                            adj[b].remove(a)
                            adj[c].remove(d)
                            adj[d].remove(c)
                            adj[a].add(c)
                            adj[c].add(a)
                            adj[b].add(d)
                            adj[d].add(b)
                            
                            edges = new_edges
                            
                            # Re-check girth
                            if check_girth_fast(adj, n, g):
                                best_n = n
                                best_edges = edges
                                found = True
                                improved = True
                                break
                            
                            # Update cycle info? 
                            # The cycle might have broken or changed.
                            # Re-find short cycle
                            short_len, cycle_nodes = find_short_cycle(adj, n, g)
                            if short_len >= g:
                                break # Should have been caught above
                            if cycle_nodes is None:
                                break
                            cycle_edges = get_cycle_edges(cycle_nodes, adj)
                    
                    if found:
                        break
        
            if found:
                # We found a graph with n = 10*L.
                # Since we iterate L upwards, this is the smallest n found so far.
                # But maybe a smaller L works with more attempts?
                # We started at L=94. If we found one, it's likely close to optimal.
                # Let's break and return.
                break
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback to LUW
        return luw_construct(k, g)
    
    # For other (k, g), use LUW
    return luw_construct(k, g)
