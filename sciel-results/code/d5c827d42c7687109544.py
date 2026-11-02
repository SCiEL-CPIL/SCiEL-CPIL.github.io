import time
import sys
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore bound for (k, g)
    # For even g: n >= 2k * ((k-1)^(g/2 - 1) - 1) / (k-2)
    if g % 2 == 0:
        moore_bound = 2 * k * ((k - 1) ** (g // 2 - 1) - 1) // (k - 2)
    else:
        moore_bound = 1 + k * ((k - 1) ** ((g - 1) // 2 - 1) * (k - 1) + 1) // (k - 2) if k > 2 else g
    
    # For k=4, g=10, moore_bound = 320.
    
    # Strategy: Search for a circulant graph of size n >= moore_bound
    # A circulant graph on n vertices is defined by a set of offsets S.
    # For k=4, |S|=2, S={a, b}.
    # Edges: (i, i+a), (i, i-a), (i, i+b), (i, i-b) mod n.
    
    def check_girth_circulant(n, a, b):
        # Returns girth of the circulant graph C(n, {a, b})
        # Girth is the length of the shortest cycle.
        # We check for cycles of length < g (which is 10).
        # So we need to ensure no cycle of length 3, 4, 5, 6, 7, 8, 9.
        
        # Optimization: If a or b is 0, invalid.
        if a == 0 or b == 0:
            return 0
        # If a == b or a == -b (mod n), multi-edge or loop.
        if a % n == b % n or a % n == (-b) % n:
            return 0
        
        # BFS from each vertex to find shortest cycle
        # State: (node, prev_node) to enforce non-backtracking
        # Actually, for girth, we just need the shortest simple cycle.
        # In a 4-regular graph, backtracking is the main source of short "cycles" in walks.
        # Non-backtracking walk: cannot go back to the node we came from.
        
        # We can run BFS on the state space (v, u) where u is the previous vertex.
        # Start: (v, -1) for all v? No, start from a single v, but since it's vertex transitive,
        # checking from v=0 is sufficient for the girth?
        # Yes, in a vertex-transitive graph, the girth is the same from all vertices.
        # Circulant graphs are vertex-transitive.
        
        # So we just run BFS from 0.
        # State: (current, previous)
        # Visited: set of (current, previous) pairs?
        # Or just store the distance to (current, previous).
        
        # However, a cycle is formed when we reach a state (u, v) where (v, u) was visited?
        # No.
        # Standard approach:
        # BFS from 0.
        # dist[(v, prev)] = d
        # If we reach a node x that we have seen before in a different state, we have a cycle.
        # Specifically, if we are at (x, y) and we have visited (x, z) with z != y,
        # then we have a cycle of length dist[(x, y)] + dist[(x, z)] + 1?
        # Not exactly.
        
        # Simpler:
        # Since n is small (<= 400), we can just do a standard BFS on the graph
        # but keep track of the parent pointer to detect cycles.
        # However, standard BFS finds shortest paths, not necessarily shortest cycles.
        # To find girth, we can do:
        # For each edge (u, v), remove it, find shortest path from u to v, add 1.
        # This is O(E * (V+E)).
        # E = 2n. V = n.
        # O(n^3). For n=320, 320^3 = 32 million.
        # 50,000 pairs * 32 million = 1.6e12. Too slow.
        
        # Better:
        # BFS from each vertex v, keeping track of the depth and the parent.
        # If we visit a node u that is already in the BFS tree, and u is not the parent of current,
        # we have a cycle.
        # The length is depth[current] + depth[u] + 1.
        # We take the minimum over all such encounters.
        
        # Since it's vertex transitive, we only need to run this from v=0?
        # NO. The shortest cycle might not pass through 0.
        # But in a vertex-transitive graph, the girth is the same.
        # Wait, the girth is a property of the graph.
        # If the graph is vertex-transitive, every vertex is "the same".
        # So the shortest cycle through 0 is the girth?
        # Yes, because if there is a cycle of length L not through 0,
        # by symmetry, there is a cycle of length L through 0.
        
        # So we only need to check cycles through 0.
        # This reduces the work significantly.
        
        # BFS from 0.
        # dist[v] = shortest distance from 0 to v.
        # parent[v] = previous node in BFS tree.
        # If we encounter a visited node u (not parent), we have a cycle.
        # But BFS tree is unique for shortest paths.
        # We need to consider all paths?
        # No, the shortest cycle through 0 is found by:
        # For each neighbor v of 0, find shortest path from v to 0 in G \ {0}?
        # Or simpler:
        # Run BFS from 0.
        # When we process a node u, for each neighbor w:
        #   If w is visited and w is not parent[u]:
        #      cycle_len = dist[u] + dist[w] + 1
        #   If w is not visited:
        #      dist[w] = dist[u] + 1
        #      parent[w] = u
        
        # This finds the shortest cycle through 0?
        # Yes, because any cycle through 0 can be decomposed into two paths from 0.
        # The shortest such cycle will be detected when the second path meets the first.
        
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        queue = [0]
        head = 0
        min_cycle = float('inf')
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            # Neighbors: u+a, u-a, u+b, u-b
            neighbors = [(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]
            
            for w in neighbors:
                if w == parent[u]:
                    continue
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                else:
                    # Back edge
                    cycle_len = dist[u] + dist[w] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
        
        return min_cycle

    # Search for n
    # Start from moore_bound
    # Limit n to moore_bound + 50 to keep time manageable
    max_n = moore_bound + 50
    
    for n in range(moore_bound, max_n + 1):
        if time.time() > deadline - 5:
            break
            
        # Iterate over pairs (a, b)
        # a in 1..n//2, b in a+1..n//2
        # Pruning: gcd(a, b, n) == 1
        for a in range(1, n // 2 + 1):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, n // 2 + 1):
                if time.time() > deadline - 2:
                    break
                
                # Check connectivity (gcd)
                if math.gcd(math.gcd(a, b), n) != 1:
                    continue
                
                # Check multi-edges
                if a % n == b % n or a % n == (-b) % n:
                    continue
                
                girth = check_girth_circulant(n, a, b)
                if girth >= g:
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i - a) % n))
                        edges.append((i, (i + b) % n))
                        edges.append((i, (i - b) % n))
                    
                    # Remove duplicates and reverse edges to ensure simple graph representation
                    # The task asks for a list of edges (i, j).
                    # Usually (i, j) with i < j is preferred, but the evaluator might accept any.
                    # "list of edges (i, j) on vertices 0..n-1"
                    # "simple (no loops or repeated edges)"
                    # My list has (i, j) and (j, i) potentially.
                    # Let's filter to i < j.
                    unique_edges = set()
                    for u, v in edges:
                        if u < v:
                            unique_edges.add((u, v))
                        else:
                            unique_edges.add((v, u))
                    
                    return n, list(unique_edges)
        
    # Fallback: LUW Construction (from seed)
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
