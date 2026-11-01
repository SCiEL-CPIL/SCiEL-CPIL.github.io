import time
import random
import heapq
import sys

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _generate_random_regular(n, k):
    """Generate a random simple k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None
    # Configuration model
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    while True:
        random.shuffle(half_edges)
        edges = set()
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            edges.add((u, v) if u < v else (v, u))
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return edges, adj
        # If not valid, retry. For large n, this might take a while.
        # We can use 2-switches to fix it, but for simplicity, retrying is okay for n~1000 if k is small.
        # Actually, probability of self-loop is 1/n. Probability of multi-edge is higher.
        # For n=1000, k=5, it should converge quickly.

def _compute_girth(n, adj):
    """Compute the girth of the graph."""
    min_girth = float('inf')
    for start in range(n):
        # BFS from start
        visited = [-1] * n
        visited[start] = 0
        queue = [start]
        # We only care about girth < 9, so we can stop early?
        # No, we need the exact min girth to know if we reached 9.
        # But if we find a cycle of length 3, we know min_girth <= 3.
        # We can stop BFS for this start if we found a cycle of length 3.
        # But we need to find *a* cycle.
        # Standard BFS for girth:
        # For each start, BFS. If we find an edge (u, v) where visited[u] == visited[v] - 1? No.
        # If we find a neighbor w of u such that visited[w] != -1 and w != parent[u], then cycle length = visited[u] + visited[w] + 1.
        
        # Optimization: If we find a cycle of length L, and L < min_girth, update.
        # If L == 3, we can break out of BFS for this start? No, other starts might have 3 too, but we already have 3.
        # We can break out of the *outer* loop if min_girth == 3? No, we need to know if it's 3.
        # If we just want to know if girth >= 9, we can stop as soon as we find a cycle < 9.
        
        # Let's just compute min girth up to 8.
        # If no cycle < 9 found, girth >= 9.
        
        # We will use a BFS that stops when depth reaches current min_girth - 1.
        
        # Actually, simpler:
        # For each start, do BFS.
        # Keep track of parent.
        # If we encounter a visited node that is not the parent, we found a cycle.
        # Length = dist[u] + dist[v] + 1.
        
        # To speed up, we can stop BFS at depth 4 (since we care about cycles up to 8).
        # If no cycle found up to depth 4, then no cycle < 9?
        # A cycle of length 8 involves nodes at distance 4 from each other.
        # So BFS depth 4 is sufficient to detect cycles of length <= 9?
        # Cycle length $L$. Nodes $u, v$ in cycle. dist(u, start) + dist(v, start) + 1 = L? No.
        # In BFS tree, if edge (u,v) exists and u,v are in tree, cycle length = dist[u] + dist[v] + 1.
        # If L <= 8, then dist[u] + dist[v] <= 7.
        # So max(dist[u], dist[v]) <= 7.
        # So BFS depth 7 is sufficient to find all cycles of length <= 8?
        # Yes.
        
        max_depth = 7
        for i in range(len(queue)):
            u = queue[i]
            if visited[u] >= max_depth:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # Ensure we don't count the tree edge
                    # If v is parent of u, skip. We don't store parent, but we can check dist.
                    # If visited[v] == visited[u] - 1, it's likely the parent.
                    # But in undirected graph, if v is parent, we visited v before u.
                    # So visited[v] < visited[u].
                    # If visited[v] == visited[u] - 1, it's the tree edge.
                    # If visited[v] >= visited[u], it's a cross edge (back edge in undirected).
                    # Wait, in undirected BFS, non-tree edges connect nodes at same or adjacent levels.
                    # If visited[v] == visited[u], cycle length = 2*visited[u] + 1.
                    # If visited[v] == visited[u] - 1, it could be parent or cross.
                    # To be safe, we can store parent.
                    
                    # Let's just compute the length if it looks like a cycle.
                    # If v != u's parent.
                    # We'll just calculate L = visited[u] + visited[v] + 1.
                    # If L < min_girth and L < 9:
                    #    min_girth = L
                    #    if min_girth == 3: return 3
                    #    break # No, other cycles might be smaller? No, 3 is min.
                    
                    # We'll refine this in the code.
                    pass
                    
    # This BFS is getting complex. Let's use a simpler, slightly slower but correct method.
    # Since n=1116 is small, we can afford a bit more overhead.
    
    # Actually, we can just use the property: Girth >= 9 iff no cycles of length 3,4,5,6,7,8.
    # We can check for existence of these cycles.
    
    return min_girth

def _try_switching(n, k, g, time_limit=100.0):
    start_time = time.time()
    
    # 1. Generate initial graph
    res = _generate_random_regular(n, k)
    if res is None:
        return None
    edges, adj = res
    
    # 2. Check Girth
    # We need a fast girth check.
    # We'll implement a function to find a short cycle if one exists.
    
    def find_short_cycle(adj, max_len=8):
        # Returns (cycle_nodes, length) or None
        n = len(adj)
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            found = False
            while queue and not found:
                u = queue.pop(0)
                if dist[u] >= max_len:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        # Check if it's a valid cycle (not just the tree edge)
                        # In undirected graph, if dist[v] == dist[u] - 1 and parent[u] == v, it's tree edge.
                        # If dist[v] == dist[u] + 1 and parent[v] == u, it's tree edge (reverse).
                        # Otherwise, it's a back/cross edge.
                        if v == parent[u]:
                            continue
                        # Cycle length
                        # Path start -> ... -> u -> v -> ... -> start?
                        # No, BFS tree is rooted at start.
                        # The cycle is start -> ... -> v -> u -> ... -> start?
                        # Actually, the cycle is formed by the path start->v, path start->u, and edge (u,v).
                        # Length = dist[u] + dist[v] + 1.
                        L = dist[u] + dist[v] + 1
                        if L < max_len: # We want < 9, so <= 8
                            # Reconstruct cycle
                            # Path from u to start
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                if cur == start: break
                                cur = parent[cur]
                            # Path from v to start
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                if cur == start: break
                                cur = parent[cur]
                            
                            # The cycle is path_u (reversed?) + path_v?
                            # Actually, we just need the edges to switch.
                            # We can just return the edge (u,v) and the fact that a short cycle exists.
                            # For switching, we just need to pick an edge in the cycle.
                            # (u,v) is in the cycle.
                            return (u, v, L)
            if found:
                break
        return None

    current_girth = 9 # Assume we want to verify if it's >= 9
    # Check if current graph has girth >= 9
    if find_short_cycle(adj, 8) is None:
        # Girth >= 9
        edge_list = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edge_list.append((i, j))
        return n, edge_list

    # 3. Switching Loop
    # We try to increase girth.
    # Strategy: Find a short cycle, pick an edge in it, try to switch it.
    
    attempts = 0
    max_attempts = 500000 # Adjust based on time
    
    while time.time() - start_time < time_limit:
        # Find a short cycle
        cycle_info = find_short_cycle(adj, 8)
        if cycle_info is None:
            # Girth >= 9
            edge_list = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edge_list.append((i, j))
            return n, edge_list
            
        u, v, L = cycle_info
        
        # Try to find a switch that breaks this cycle
        # Pick a random edge (x, y) to switch with (u, v)
        # We need x, y such that (x,y) is an edge, x!=u, x!=v, y!=u, y!=v
        # And (u,x), (v,y) are not edges.
        
        # To speed up, we can pick x from adj[u] (excluding v) and y from adj[x] (excluding u)?
        # No, we are replacing (u,v) and (x,y).
        
        # Let's pick x randomly from 0..n-1? No, x must be connected to some y.
        # Let's pick a random edge (x,y) from the graph.
        # We can maintain a list of edges.
        
        # For simplicity, we pick x from adj[u] (excluding v)?
        # If we pick x from adj[u], then (u,x) is already an edge. We can't add it.
        # We need to add (u,x) and (v,y).
        # So x must NOT be in adj[u].
        # And y must NOT be in adj[v].
        
        # This is restrictive.
        # Alternative:
        # Pick x randomly.
        # Pick y from adj[x] (excluding u).
        # Check if (v,y) is not an edge.
        # Check if (x,y) is an edge (yes, by choice).
        # Check if (u,v) is an edge (yes).
        # Check if x != u, y != v, etc.
        
        # Let's try this.
        success = False
        for _ in range(100): # 100 attempts per cycle found
            x = random.randint(0, n-1)
            if x == u or x == v:
                continue
            if not adj[x]:
                continue
            # Pick y from adj[x]
            y = random.choice(list(adj[x]))
            if y == u or y == v:
                continue
            if y == x:
                continue
            
            # Check if (v,y) is an edge
            if y in adj[v]:
                continue
            # Check if (u,x) is an edge
            if x in adj[u]:
                continue
            
            # Perform switch: remove (u,v), (x,y); add (u,x), (v,y)
            # Check simplicity
            # (u,x) and (v,y) are new.
            # We must ensure they don't create self loops (x!=u, y!=v checked).
            
            # Apply
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            # Check if this switch helped
            # We don't need to recompute full girth immediately, just check if the specific cycle is broken?
            # No, it might create a new short cycle.
            # We just continue the loop. The next iteration will find a short cycle if one exists.
            # If no short cycle exists, we return.
            
            success = True
            break
            
        if not success:
            # Try a different cycle or restart?
            # We can just continue.
            pass
            
    # If time runs out, return None
    return None

def construct(k, g):
    if k == 5 and g == 9:
        target_ns = [1116, 1120, 1150, 1200]
        time_per_attempt = 25.0
        
        for n in target_ns:
            res = _try_switching(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
