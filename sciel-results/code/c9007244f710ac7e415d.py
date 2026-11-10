import time
import random
from collections import deque

def construct(k, g):
    if k != 3:
        return _lww_fallback(k, g)
    
    # Calculate LWW size for comparison
    n_lww = _get_lww_size(3, g)
    
    deadline = time.time() + 110
    
    # Try to find a smaller graph
    best_n = n_lww
    best_edges = None
    
    # Run a few attempts
    while time.time() < deadline:
        n, edges = _backtrack_cage(3, g, deadline)
        if n < best_n:
            best_n = n
            best_edges = edges
            # If we get very close to Moore bound, stop? 
            # Moore bound for k=3, g=16 is 382.
            # If we find < 500, that's great.
            if best_n < 500:
                break
        # If we didn't improve, maybe restart with different random seed
        # The function _backtrack_cage uses random internally?
        # Let's ensure it does.
        
    if best_edges is not None:
        return best_n, best_edges
    else:
        return _lww_fallback(3, g)

def _get_lww_size(k, g):
    q = k
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)
    else:
        d = max(4, g - 4)
    return q ** d

def _backtrack_cage(k, g, deadline):
    random.seed()
    adj = [[] for _ in range(1)] # Start with vertex 0
    
    # Add first k neighbors to vertex 0
    for i in range(1, k+1):
        adj.append([])
        adj[0].append(i)
        adj[i].append(0)
    
    n = k + 1
    
    # Queue of vertices that need edges
    # We prioritize vertices with lowest degree
    import heapq
    # Heap of (degree, vertex)
    heap = [(len(adj[i]), i) for i in range(n)]
    heapq.heapify(heap)
    
    def can_add_edge(u, v):
        if u == v: return False
        if v in adj[u]: return False
        # BFS from u to depth g-2
        # If v is found, return False
        dist = {u: 0}
        queue = deque([u])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= g - 2:
                continue
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = d + 1
                    if nb == v:
                        return False
                    queue.append(nb)
        return True

    # Main loop
    while time.time() < deadline:
        if not heap:
            break
        deg, v = heapq.heappop(heap)
        if deg >= k:
            continue
            
        # We need to add k - deg edges to v
        # Try to find existing vertices to connect to
        candidates = [u for u in range(n) if u != v and len(adj[u]) < k and can_add_edge(v, u)]
        
        if not candidates:
            # Add new vertices
            for _ in range(k - deg):
                new_v = len(adj)
                adj.append([])
                adj[v].append(new_v)
                adj[new_v].append(v)
                heapq.heappush(heap, (1, new_v))
                n += 1
            # v is now complete
            continue
        else:
            # Pick one candidate
            # Heuristic: pick one with highest distance?
            # For now, pick random
            u = random.choice(candidates)
            adj[v].append(u)
            adj[u].append(v)
            
            # Update heap
            heapq.heappush(heap, (len(adj[v]), v))
            heapq.heappush(heap, (len(adj[u]), u))
            
    # Check if complete
    for i in range(n):
        if len(adj[i]) != k:
            return n_lww, None # Fail, return LWW size marker
            
    # Verify girth
    # Simple check: BFS from every node? Too slow for 1000 nodes?
    # 1000 * 3^7 is ok.
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    if dist[nb] < g and nb == start: # Cycle back to start
                        # Wait, this detects cycles through start.
                        # A cycle of length L means dist(start, start) via this path is L.
                        # But BFS finds shortest path.
                        # If shortest path from start to start is < g, we have a short cycle.
                        # But dist[start] is 0.
                        # We need to check if there is a path of length < g from start to start.
                        # This is equivalent to checking if any neighbor of start is reachable from start in < g-1 steps.
                        pass
                    
    # Better girth check:
    # For each edge (u,v), remove it, find dist(u,v). If dist < g-1, cycle length < g.
    # This is O(E * BFS).
    # E = 1.5 * n.
    # If n=1000, E=1500. BFS is ~1000. Total 1.5M ops. Fast.
    
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Temporarily remove edge
                adj[u].remove(v)
                adj[v].remove(u)
                
                # BFS from u to v
                dist = {u: 0}
                q = deque([u])
                found_dist = -1
                while q:
                    node = q.popleft()
                    if node == v:
                        found_dist = dist[node]
                        break
                    for nb in adj[node]:
                        if nb not in dist:
                            dist[nb] = dist[node] + 1
                            q.append(nb)
                            
                # Restore edge
                adj[u].append(v)
                adj[v].append(u)
                
                if found_dist != -1 and found_dist + 1 < g:
                    return n_lww, None # Invalid girth
                    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                
    return n, edges

def _lww_fallback(k, g):
    q = k
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)
    else:
        d = max(4, g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn = lambda x, y: x ^ y
        sub_fn = lambda x, y: x ^ y
        mul_fn = lambda x, y: table[x][y]
    else:
        add_fn = lambda x, y: (x + y) % q
        sub_fn = lambda x, y: (x - y) % q
        mul_fn = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
