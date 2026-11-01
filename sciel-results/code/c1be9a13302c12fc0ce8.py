import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110  # 10s margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Try to find a smaller graph using local search
    # Target: 4-regular, girth >= 10
    # We assume k=4, g=10 based on the task description, but handle generally?
    # The task says "evaluator passes this task's degree k and girth g".
    # So k and g are inputs.
    
    if k != 4 or g != 10:
        # Fallback for other parameters
        return luw_construct(k, g)

    # Local search for k=4, g=10
    best_n, best_edges = None, None
    
    # We know 320 is the best known. Try to find it or better.
    # Start with N close to Moore bound?
    # Moore bound is 485.
    # Try N from 300 to 400.
    
    def is_valid_graph(n, edges, k, g):
        # Check degree
        deg = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if u == v: return False
            if deg[u] >= k or deg[v] >= k: return False # Simple check, but we generate regular
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        for d in deg:
            if d != k: return False
        
        # Check girth
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for i in range(len(queue)):
                u = queue[i]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Back edge or cross edge
                        # Cycle length = dist[u] + dist[v] + 1? No, if u-v is edge.
                        # If v is already visited, the cycle is u-v-...-u.
                        # Length is dist[u] + dist[v] + 1?
                        # Actually, if we find an edge (u,v) where dist[v] != dist[u]+1 and dist[u] != dist[v]+1,
                        # or if it's a back edge to an ancestor.
                        # Simpler: just check if any cycle < g is found.
                        # In BFS, the first time we see a visited node, we can compute cycle length.
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def generate_random_bipartite_4_regular(n):
        # n must be even
        half = n // 2
        # A = 0..half-1, B = half..n-1
        # Each node in A connects to 4 distinct nodes in B.
        # Each node in B connects to 4 distinct nodes in A.
        # This is equivalent to a 4-regular bipartite graph.
        # We can construct this by taking 4 perfect matchings.
        edges = []
        # Generate 4 random permutations of B for each A? No, that would make B nodes have degree 4 if each A picks distinct B's?
        # No, if we just pick 4 random B's for each A, the degrees of B are not guaranteed to be 4.
        # We need a 4-regular bipartite graph.
        # Algorithm:
        # 1. Create a list of 4 * half "slots" in A and 4 * half "slots" in B.
        # 2. Randomly pair A-slots with B-slots.
        # 3. This ensures each A-node has 4 edges and each B-node has 4 edges.
        # 4. Check for multi-edges. If (a,b) appears more than once, reshuffle.
        
        a_slots = [i for i in range(half) for _ in range(4)]
        b_slots = [i + half for i in range(half) for _ in range(4)]
        
        for _ in range(100): # Max attempts
            random.shuffle(b_slots)
            edge_set = set()
            valid = True
            for a, b in zip(a_slots, b_slots):
                if (a, b) in edge_set:
                    valid = False
                    break
                edge_set.add((a, b))
            if valid:
                return list(edge_set)
        return None

    def get_girth(n, adj, g):
        # Returns min cycle length found, or g if none < g
        min_len = g
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for i in range(len(queue)):
                u = queue[i]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            # Cycle
                            # We need the length of the cycle formed by u-v and the path in the tree.
                            # This is tricky to compute exactly without LCA.
                            # But for small g, we can just do BFS up to depth g//2.
                            pass
        # Simpler girth check: BFS from each node up to depth (g-1)//2
        # If we find a node at depth d that was already reached at depth d' where d+d' < g, then cycle < g.
        # Actually, if we do BFS from each node, and we find an edge (u,v) such that dist[u] + dist[v] + 1 < g, 
        # that's not quite right.
        
        # Correct Girth Check:
        # For each edge (u,v), the girth is the min over all edges of the length of the shortest cycle containing it.
        # Or simply: BFS from each vertex. If we encounter a visited vertex, we can calculate cycle length.
        # To be safe and fast for g=10:
        # BFS from each vertex to depth 4.
        # If any vertex is reached by two different paths of length <= 4, we have a cycle of length <= 8?
        # No, if u is reached at depth 2 via p1 and depth 2 via p2, cycle is 4.
        # If u is reached at depth 3 via p1 and depth 3 via p2, cycle is 6.
        # If u is reached at depth 4 via p1 and depth 4 via p2, cycle is 8.
        # We need to check if any cycle < 10 exists.
        # So we check for cycles of length 4, 6, 8.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for i in range(len(queue)):
                u = queue[i]
                if dist[u] >= 4: # We only care about cycles up to 8, so depth 4 is enough to detect them?
                    # If a cycle of length 8 exists, it has a diameter 4?
                    # Yes, a cycle of length 2k has diameter k.
                    # So BFS to depth 4 from any vertex in an 8-cycle will find the other side.
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            # Found a cycle
                            # Length is dist[u] + dist[v] + 1?
                            # This formula works if the tree paths don't overlap too much?
                            # Actually, for simple cycles, if we just want to know IF a short cycle exists:
                            # If dist[u] + dist[v] + 1 < g, is it a valid cycle?
                            # Not necessarily. The paths might share nodes.
                            # However, for girth checking in small graphs, this heuristic often works or we can just use a standard algorithm.
                            # Let's use a more robust check: 
                            # Just do BFS and if we find ANY back edge, we can trace back to find the cycle length.
                            # But for speed, if dist[u] + dist[v] + 1 < g, it's a strong candidate.
                            if dist[u] + dist[v] + 1 < g:
                                return dist[u] + dist[v] + 1
        return g

    # Main Loop
    for n_try in range(300, 400, 10):
        if time.time() > deadline:
            break
        # Try to find a graph of size n_try
        for attempt in range(20):
            if time.time() > deadline:
                break
            edges = generate_random_bipartite_4_regular(n_try)
            if not edges:
                continue
            
            # Build adjacency
            adj = [[] for _ in range(n_try)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check girth
            girth = get_girth(n_try, adj, g)
            
            if girth >= g:
                return n_try, edges
            
            # Local Search
            current_girth = girth
            for step in range(1000):
                if time.time() > deadline:
                    break
                # Pick random edge
                u, v = random.choice(edges)
                # Find another edge (x, y)
                x, y = random.choice(edges)
                if (u,v) == (x,y) or (u,v) == (y,x):
                    continue
                # Check if swap is valid
                # Swap (u,v) and (x,y) with (u,y) and (v,x)
                # Must ensure (u,y) and (v,x) are not existing edges
                if (u, y) in adj[u] or (v, x) in adj[v]:
                    continue
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].append(y)
                adj[y].append(u)
                adj[v].append(x)
                adj[x].append(v)
                
                # Update edges list
                # Remove old, add new
                # For simplicity, rebuild edges from adj? No, too slow.
                # Just keep a set of edges for checking.
                edge_set = set()
                for i in range(n_try):
                    for j in adj[i]:
                        if i < j:
                            edge_set.add((i,j))
                        else:
                            edge_set.add((j,i))
                
                new_girth = get_girth(n_try, adj, g)
                
                if new_girth >= g:
                    return n_try, list(edge_set)
                
                # Accept if better, or randomly (Simulated Annealing)
                if new_girth > current_girth:
                    current_girth = new_girth
                elif random.random() < 0.1: # Random restart/accept
                    pass # Keep it for now? Or revert?
                    # Let's keep it to explore landscape
                else:
                    # Revert
                    adj[u].remove(y)
                    adj[y].remove(u)
                    adj[v].remove(x)
                    adj[x].remove(v)
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[x].append(y)
                    adj[y].append(x)

    # Fallback
    return luw_construct(k, g)
