import random
import time
from collections import deque

def construct(k, g):
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

    # Moore bound
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        
    # Current best known for k=4, g=10 is 320.
    # We try to find a graph with n < 320.
    # Search range: moore to 319.
    
    start_time = time.time()
    timeout = 110 # Leave 10s for fallback and overhead
    
    # Helper to check girth of a graph given adjacency list
    def check_girth(adj, g_target):
        n = len(adj)
        # BFS from each node to find shortest cycle
        # Optimization: If we find a cycle of length < g_target, return False
        # We can stop early if we find any such cycle.
        
        # For speed, we can just run BFS from each node up to depth g_target//2 - 1
        # and check for back edges.
        
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if visited[u] >= g_target // 2:
                    continue
                    
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < g_target:
                            return False
            # If we finish BFS from start without finding short cycle, 
            # it doesn't mean global girth is ok, but if we find one, we fail.
            # Actually, if we don't find a short cycle from start, there might still be one elsewhere.
            # So we must check all starts? 
            # Yes, girth is global.
            # However, if the graph is random, checking a few nodes might suffice? No, must be exact.
            # But we can short-circuit: if we find a short cycle, return False immediately.
            # If we check all nodes and find none, girth >= g_target.
            
        return True

    # Helper to generate a random k-regular graph using Configuration Model
    def gen_random_k_regular(n, k):
        # Create list of stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Shuffle
        random.shuffle(stubs)
        
        # Pair them up
        edges = set()
        adj = [[] for _ in range(n)]
        
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Loop
            if (min(u,v), max(u,v)) in edges:
                return None # Multi-edge
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
            
        return adj, list(edges)

    # Search for n < 320
    # We try from lower n upwards to maximize score improvement
    # But higher n is easier to find.
    # Let's try a range.
    # Moore bound for 4,10 is 242.
    # Try 243 to 319.
    
    # To be efficient, we can try random n in that range or iterate.
    # Iterating from 319 down to 243 might be safer as larger n is easier to satisfy girth constraint.
    # But we want smallest n.
    # Let's try a few random samples for each n, or iterate n and try limited samples.
    
    # Strategy:
    # 1. Try n from 243 to 319.
    # 2. For each n, try up to 50 random graphs.
    # 3. If found, return.
    
    # Actually, 50 * 77 = 3850 graphs.
    # Girth check is O(N^2) roughly? No, BFS is O(N+E). N=300, E=600. Very fast.
    # 3850 * 1000 ops = 4M ops. Should be fast in Python?
    # Python loops are slow. 3850 BFS runs might take a few seconds.
    # Let's optimize girth check.
    
    # Better girth check:
    # We only need to know if girth < g.
    # We can do BFS from each node, but stop if depth > g//2.
    
    def has_short_cycle(adj, g_limit):
        n = len(adj)
        max_depth = g_limit // 2 - 1
        for start in range(n):
            # BFS from start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] > max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Check if this edge creates a cycle shorter than g_limit
                        # The cycle length is dist[u] + dist[v] + 1
                        # But we need to ensure it's a simple cycle.
                        # In BFS tree, back edge (u,v) creates a cycle.
                        # Length = dist[u] + dist[v] + 1.
                        # Is this always the shortest cycle involving this edge?
                        # Yes, for the path from start.
                        # But we need global girth.
                        # If dist[u] + dist[v] + 1 < g_limit, we found a short cycle.
                        if dist[u] + dist[v] + 1 < g_limit:
                            return True
        return False

    # Try to find a graph
    # We'll iterate n from 243 to 319.
    # If we find one, we return it.
    
    # To save time, we can randomize the order or start from a promising range.
    # Known best is 320.
    # Let's try 243 to 319.
    
    # Limit attempts per n to avoid timeout.
    attempts_per_n = 20
    
    for n in range(243, 320):
        if time.time() - start_time > timeout:
            break
            
        for _ in range(attempts_per_n):
            if time.time() - start_time > timeout:
                break
                
            res = gen_random_k_regular(n, k)
            if res is None:
                continue
                
            adj, edges = res
            
            # Check girth
            if not has_short_cycle(adj, g):
                # Valid graph found
                return n, edges

    # Fallback
    return luw_construct(k, g)
