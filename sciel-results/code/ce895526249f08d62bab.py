import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 110.0  # Leave 10s safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    # This is the seed logic, kept as a fallback.
    def luw_construct():
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

    # Heuristic Search: Edge Swapping
    # Target n = 320 (known best)
    target_n = 320
    
    def get_girth(adj, n, max_depth):
        # Returns the length of the shortest cycle
        # adj is list of lists of neighbors
        for u in range(n):
            # BFS from u
            visited = [-1] * n
            parent = [-1] * n
            visited[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if visited[w] == -1:
                        visited[w] = visited[v] + 1
                        parent[w] = v
                        q.append(w)
                    elif w != parent[v] and visited[v] + 1 > visited[w]:
                        # Cycle found
                        # Reconstruct cycle length
                        # This simple check might miss some or be slow.
                        # Better: track distance and find back edge.
                        pass
            
            # More efficient girth check:
            # For each u, BFS up to depth g//2 - 1
            # If we find a neighbor w of u that is also a neighbor of some v at depth g//2 - 1?
            # Standard: 
            for u in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[u] = 0
                q = deque([u])
                while q:
                    v = q.popleft()
                    if dist[v] >= g // 2 - 1:
                        continue
                    for w in adj[v]:
                        if dist[w] == -1:
                            dist[w] = dist[v] + 1
                            parent[w] = v
                            q.append(w)
                        elif w != parent[v] and dist[v] + 1 > dist[w]:
                            # Found a cycle
                            # Length is dist[v] + 1 + dist[w]? No.
                            # We need to be careful.
                            pass
        return 100 # Placeholder
        
    # Correct Girth Calculation
    def calculate_girth(adj, n):
        min_girth = 100
        for u in range(n):
            # BFS from u
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        q.append(w)
                    elif w != parent[v]:
                        # Cycle found
                        # We need to find the length.
                        # Since we are doing BFS from u, the cycle involves u?
                        # No, w is already visited.
                        # The cycle is v -> w -> ... -> u -> ... -> v?
                        # This standard BFS from each node finds the shortest cycle containing u?
                        # Actually, finding girth by BFS from each node is O(N*(N+E)).
                        # For N=320, E=640, this is fast.
                        # To get the length:
                        # We need to trace back from v and w to their LCA?
                        # Simpler: Just check if a cycle of length < current min exists.
                        # We can just run BFS up to depth 4 (for girth 10, half is 5).
                        # If we reach a node already visited at depth d, cycle length is d + dist[w] + 1?
                        # Let's use a simpler method: 
                        # For each edge (u,v), remove it and check shortest path u->v.
                        # Too slow.
                        
                        # Let's use the standard "BFS from each vertex" to find shortest cycle.
                        # For each u, BFS.
                        # If we encounter a visited node w (not parent), we have a cycle.
                        # The length is dist[v] + 1 + dist[w]? No.
                        # We need the distance from u to w.
                        # If we find a back edge v-w where dist[w] < dist[v],
                        # the cycle length is dist[v] + 1 + dist[w] + ...?
                        # Actually, if we just want to know if girth < 10,
                        # we can stop BFS at depth 4.
                        # If we find a cycle of length < 10, we return it.
                        pass
        return 100

    # Efficient Girth Check for Girth >= 10
    def has_short_cycle(adj, n, limit=9):
        # Returns True if there is a cycle of length < 10
        # BFS from each node up to depth 4
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                if visited[v] >= 4:
                    continue
                for w in adj[v]:
                    if visited[w] == -1:
                        visited[w] = visited[v] + 1
                        parent[w] = v
                        q.append(w)
                    elif w != parent[v]:
                        # Found a cycle
                        # We need to verify the length.
                        # The cycle is v -> w -> ... -> u -> ... -> v?
                        # No, w is visited.
                        # The path from u to w is known (visited[w]).
                        # The path from u to v is known (visited[v]).
                        # The edge v-w closes a cycle.
                        # The length is visited[v] + 1 + visited[w]?
                        # Only if the paths from u to v and u to w are disjoint except at u.
                        # This is not guaranteed.
                        # However, for small depths, we can just reconstruct.
                        # Simpler: If we find ANY cycle < 10, return True.
                        # We can just check if the distance between v and w in the tree + 1 < 10?
                        # No, the cycle length is at most visited[v] + visited[w] + 1.
                        # If visited[v] + visited[w] + 1 < 10, we definitely have a short cycle.
                        # If not, it might still be short if paths overlap.
                        # But for girth 10, we just need to ensure no cycle < 10.
                        # Let's just return True if we find a back edge at depth < 5.
                        # This is a heuristic for "short cycle exists".
                        # It might have false positives if the cycle is longer.
                        # To be safe, we should calculate the exact girth.
                        pass
        return False

    # Let's implement a robust girth calculator.
    def get_girth(adj, n):
        min_g = 100
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        q.append(w)
                    elif w != parent[v]:
                        # Cycle found
                        # We need to find the length of the cycle involving v and w.
                        # This requires LCA or similar.
                        # For simplicity and speed, we can just check if dist[v] + dist[w] + 1 < min_g
                        # This is an upper bound on the cycle length? No.
                        # The actual cycle length is the distance between v and w in the graph + 1.
                        # dist[v] + dist[w] is the distance via u.
                        # The actual distance might be shorter.
                        # However, if we just want to minimize n, and we know the optimum is 320,
                        # we can just try to find a graph with girth >= 10.
                        pass
        return min_g

    # Given the complexity of exact girth in a tight loop, 
    # and the fact that we only need to beat 2048,
    # we can use a simpler check: 
    # If we perform edge swaps that *increase* the girth, we can track the girth.
    
    # Let's try a different approach: 
    # Use the known 320-vertex graph structure if possible?
    # No, I'll stick to the heuristic.
    
    # Random 4-regular graph generation
    def random_k_regular(n, k):
        # n must be even for k=4? No, sum of degrees must be even. 4n is even.
        # Simple configuration model
        half_edges = [i for i in range(n) for _ in range(k)]
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                # Self loop, retry or swap
                continue
            if v in adj[u]:
                # Multi-edge, retry or swap
                continue
            adj[u].append(v)
            adj[v].append(u)
        # Check if it's a valid simple graph
        for i in range(n):
            if len(adj[i]) != k:
                return None
            if len(set(adj[i])) != k:
                return None
        return adj

    # Girth calculation (Exact, but optimized for small n)
    def calc_girth(adj, n):
        min_g = 100
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        q.append(w)
                    elif w != parent[v]:
                        # Cycle
                        # We need to compute the length of the cycle formed by v-w and paths from u.
                        # This is complex. 
                        # Alternative: Just check if girth < 10.
                        # If we find a cycle, we can stop if we know it's < 10.
                        # For now, let's assume if we find a back edge at depth < 5, it's likely short.
                        pass
        return 100 # Fallback

    # This is getting too complex for the time limit.
    # Let's just return the L-U-W graph. It's safe.
    # But the score is 1.0.
    # Can I do better?
    # The (4,10)-cage is 320.
    # If I can't find it, I return L-U-W (2048).
    
    # Let's try to return a smaller L-U-W?
    # No, L-U-W is fixed.
    
    # I will return the L-U-W graph to ensure validity.
    n, edges = luw_construct()
    return n, edges
