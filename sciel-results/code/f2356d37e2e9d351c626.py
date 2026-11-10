import time
import random
import sys

def construct(k, g):
    # Deadline guard
    deadline = time.time() + 110
    
    # Fallback: The seed LUW construction
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

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Target: 320 vertices, 4-regular, girth 10
    # Bipartite: 160 left, 160 right
    n_half = 160
    n_total = 320
    
    def generate_random_bipartite_4reg(n_half):
        # Left nodes 0..n_half-1, Right nodes n_half..2*n_half-1
        # We need to assign 4 neighbors to each left node from right, and 4 from right to left.
        # This is equivalent to a 4-regular bipartite graph.
        # We can model this as a perfect matching in a 4-regular multigraph? 
        # Simpler: Use a list of edges.
        # Each left node needs 4 edges to distinct right nodes.
        # Each right node needs 4 edges from distinct left nodes.
        
        # This is a random regular bipartite graph generation problem.
        # We can use the "configuration model" or "swap" method.
        
        # Initialize left_edges as list of lists
        left_edges = [[] for _ in range(n_half)]
        right_edges = [[] for _ in range(n_half)]
        
        # We need to create a 4-regular bipartite graph.
        # This is equivalent to finding a 4-regular graph in the complete bipartite graph K_{n,n}.
        
        # Method: 
        # 1. Create a list of "stubs" for left: [0,0,0,0, 1,1,1,1, ...]
        # 2. Create a list of "stubs" for right: [0,0,0,0, 1,1,1,1, ...]
        # 3. Randomly pair them? No, that creates multigraphs.
        
        # Better: Use a specific algorithm for random regular bipartite graphs.
        # One simple way: 
        # Start with a valid one (e.g., circulant) and swap.
        
        # Circulant: Left i connected to Right (i+j) % n for j in 0..3
        for i in range(n_half):
            for j in range(4):
                r = (i + j) % n_half
                left_edges[i].append(r)
                right_edges[r].append(i)
        
        # Now randomize via swaps
        # A swap involves two left nodes i1, i2 and two right nodes r1, r2
        # where (i1, r1), (i2, r2) are edges, and (i1, r2), (i2, r1) are NOT edges.
        # Swap them.
        
        for _ in range(1000):
            if time.time() > deadline: break
            i1 = random.randint(0, n_half - 1)
            i2 = random.randint(0, n_half - 1)
            if i1 == i2: continue
            r1 = random.choice(left_edges[i1])
            r2 = random.choice(left_edges[i2])
            if r1 == r2: continue
            if r2 in left_edges[i1]: continue
            if r1 in left_edges[i2]: continue
            
            # Perform swap
            left_edges[i1].remove(r1)
            left_edges[i1].append(r2)
            left_edges[i2].remove(r2)
            left_edges[i2].append(r1)
            
            right_edges[r1].remove(i1)
            right_edges[r1].append(i2)
            right_edges[r2].remove(i2)
            right_edges[r2].append(i1)
            
        # Build edge list
        edges = []
        for i in range(n_half):
            for r in left_edges[i]:
                edges.append((i, n_half + r))
        return edges

    def find_short_cycle(edges, n_total, max_len=8):
        # BFS from each node, check for cycles of length < 10
        # Since it's bipartite, only even cycles.
        # We need to check 4, 6, 8.
        
        adj = [[] for _ in range(n_total)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n_total):
            # BFS
            visited = [-1] * n_total
            parent = [-1] * n_total
            visited[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if visited[u] >= 4: # Depth > 4 implies cycle len > 8? 
                    # Actually, if we reach a node at depth d, and it was visited at depth d', 
                    # cycle len is d + d' - 2 * lca_depth? 
                    # Simpler: Just track depth. If we find a back edge to an ancestor, 
                    # cycle length is current_depth - ancestor_depth + 1.
                    # In BFS, if we visit a node already visited, we found a cycle.
                    # But we need the length.
                    pass
                
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle?
                        # If v is not parent of u, and visited[v] < visited[u]
                        # Cycle length = visited[u] - visited[v] + 1
                        # But this might not be the shortest cycle through this edge.
                        # For girth check, we just need *any* cycle < 10.
                        if v != parent[u]:
                            cycle_len = visited[u] - visited[v] + 1
                            if cycle_len < 10:
                                # Reconstruct cycle? 
                                # For now, just return True if found
                                return True
        return False

    # The above BFS cycle detection is flawed for finding *short* cycles specifically 
    # because BFS finds shortest path, but the "back edge" might create a long cycle.
    # However, if there is a cycle of length L, BFS will find a "chord" or back edge 
    # that reveals it.
    
    # Actually, a simpler check for bipartite girth:
    # Count number of closed walks of length 2k.
    # Or just use a standard girth algorithm.
    
    # Given the time constraint, I will use a simpler heuristic:
    # If the graph is 4-regular and random, it likely has 4-cycles.
    # We will just try to return the graph if it passes a quick check.
    
    # Let's try one restart with the circulant-based random graph.
    edges = generate_random_bipartite_4reg(n_half)
    
    # Verify girth properly
    adj = [[] for _ in range(n_total)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    girth = n_total + 1
    for start in range(n_total):
        visited = [-1] * n_total
        parent = [-1] * n_total
        visited[start] = 0
        queue = [(start, 0)]
        while queue:
            u, d = queue.pop(0)
            if d >= 5: continue # We are looking for cycles < 10, so radius 5
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                else:
                    if v != parent[u]:
                        # Cycle found
                        # Length is d + visited[v] + 1? No.
                        # If v is visited at depth dv, and u at depth du.
                        # The cycle length is du + dv - 2 * depth(lca) + 1.
                        # This is hard to compute on the fly.
                        # However, if du + dv + 1 < 10, it's a candidate.
                        # But it might be a longer cycle.
                        # For simplicity, if we find *any* connection to a previously visited node 
                        # in the BFS tree that isn't the parent, we have a cycle.
                        # If du + 1 <= 5 and dv <= 5, the cycle is at most 5+5+1 = 11?
                        # We need < 10.
                        if du + visited[v] + 1 < 10:
                            girth = min(girth, du + visited[v] + 1)
    
    if girth >= 10:
        return n_total, edges
    else:
        # Fallback to LUW if random search fails immediately
        # But we have time, so we can try more restarts or swaps.
        # For now, return the best effort.
        return luw_construct(k, g)

