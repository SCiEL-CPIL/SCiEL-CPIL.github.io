import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def construct_lw(k, g):
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

    # If k != 4 or g != 10, just use LW
    if k != 4 or g != 10:
        return construct_lw(k, g)

    # Target: (4,10) cage, n=320
    # Base graph: 10 vertices, 4-regular
    # Vertices 0-9
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (0, 7), (1, 8), (2, 9), (3, 5), (4, 6)
    ]
    
    # Group G of order 32
    # Use Z_32 for simplicity in voltage assignment search
    # Or Z_2 x Z_2 x Z_2 x Z_2 x Z_2
    # Let's use Z_32
    G_size = 32
    
    def build_lift(voltages):
        # voltages: list of length 20, values in 0..31
        n = 10 * G_size
        edges = []
        # Vertex id: i * G_size + g
        for i in range(10):
            for j in range(10):
                if i < j:
                    # Find edge index
                    pass
        
        # Map edge to index
        edge_map = {}
        for idx, (u, v) in enumerate(base_edges):
            edge_map[(u, v)] = idx
            edge_map[(v, u)] = idx
            
        for u in range(10):
            for v in range(10):
                if (u, v) in edge_map:
                    e_idx = edge_map[(u, v)]
                    vol = voltages[e_idx]
                    # Lifted edge: (u, g) -- (v, g + vol)
                    # We add edge for each g
                    for g in range(G_size):
                        u_lift = u * G_size + g
                        v_lift = v * G_size + (g + vol) % G_size
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
        return n, edges

    def check_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length < min_girth
        # Non-backtracking BFS
        for start in range(n):
            # visited: dict (node, prev) -> dist
            # We want to find if there is a cycle.
            # Simpler: BFS up to depth (min_girth-1)//2
            # If we find a neighbor of start at depth d < min_girth/2, cycle exists?
            # Actually, standard girth check:
            # For each vertex, BFS. If we encounter a visited node that is not the parent, 
            # and the path length + 1 < min_girth, we have a small cycle.
            
            # Optimized: Just check if any cycle of length 3..9 exists.
            # We can use a global set of (u,v) pairs seen in walks?
            # Too slow.
            
            # Simple BFS from each vertex up to depth 4.
            # If we find a path of length L from u to v, and v is neighbor of u, 
            # then cycle length L+1.
            # We need L+1 < 10 => L < 9.
            # So check paths of length 1 to 8.
            # This is expensive.
            
            # Better: 
            # A graph has girth >= g iff it has no cycles of length < g.
            # We can check for cycles of length 3, 4, 5, 6, 7, 8, 9.
            # For length 3: triangle.
            # For length 4: square.
            # ...
            
            # Given n=320, we can do a BFS from each vertex up to depth 4.
            # If we find a back-edge to an ancestor in the BFS tree, we have a cycle.
            # Length = depth(u) - depth(v) + 1.
            # If length < 10, return False.
            
            for s in range(n):
                # BFS from s
                parent = [-1] * n
                depth = [-1] * n
                queue = [s]
                parent[s] = -2
                depth[s] = 0
                q_idx = 0
                while q_idx < len(queue):
                    u = queue[q_idx]
                    q_idx += 1
                    if depth[u] >= 4: # Max depth needed for cycle < 10 is 4?
                        # Path of length 4 from s to u.
                        # If u is neighbor of s, cycle length 5.
                        # If u is neighbor of parent[s], cycle length 4?
                        # We just need to check if any cycle < 10 is formed.
                        # Actually, if we find any cycle, we check its length.
                        pass
                    
                    for v in adj[u]:
                        if depth[v] == -1:
                            depth[v] = depth[u] + 1
                            parent[v] = u
                            if depth[v] <= 4: # Only expand if depth is small
                                queue.append(v)
                        else:
                            # Found a cycle
                            # Length is depth[u] + depth[v] - 2 * depth[lca] + 1
                            # This is complex.
                            # Simpler: If v is an ancestor of u in BFS tree?
                            # Or just: if depth[u] + 1 - depth[v] < 10 and v is not parent?
                            # Actually, any edge (u,v) where depth[u] != depth[v]+1 and depth[v] != depth[u]+1 
                            # creates a cycle of length depth[u] + depth[v] - 2*depth[lca] + 1.
                            # Since we only go to depth 4, max cycle length detected this way is small.
                            
                            # If we see an edge between two nodes at depth d1, d2 <= 4,
                            # the cycle length is at most d1 + d2 + 1 <= 9.
                            # So if we find ANY cross edge or back edge in this BFS, 
                            # the cycle is < 10.
                            
                            # Exception: The edge (u,v) is the tree edge itself.
                            if v == parent[u]:
                                continue
                            if u == parent[v]:
                                continue
                                
                            # Check if it's a cycle
                            # If depth[u] == depth[v] or |depth[u]-depth[v]| <= 1?
                            # Actually, any non-tree edge in a BFS tree of depth < 5 
                            # implies a cycle of length < 10.
                            # Why? 
                            # Max depth 4. 
                            # Path u->...->root->...->v has length <= 4+4=8.
                            # Plus edge (u,v) = 9.
                            # So any non-tree edge implies cycle <= 9.
                            return False
                # End BFS from s
        return True

    # Try to find valid voltages
    # Use random search
    found = False
    best_n = None
    best_edges = None
    
    # Fixed seed for reproducibility
    random.seed(42)
    
    num_tries = 1000
    for i in range(num_tries):
        if time.time() > deadline:
            break
        voltages = [random.randint(0, 31) for _ in range(20)]
        n, edges = build_lift(voltages)
        
        # Quick check: is it simple? Yes, by construction.
        # Check girth
        if check_girth(n, edges, 10):
            found = True
            best_n = n
            best_edges = edges
            break
            
    if found:
        return best_n, best_edges
    else:
        # Fallback to LW
        return construct_lw(k, g)

# EVOLVE-BLOCK-END
