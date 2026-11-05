import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: LUW construction
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

    if k != 4 or g != 10:
        return luw_construct()

    # Random search for 4-regular graph with girth 10
    # Target n: 242 to 350
    # We try to find the smallest n.
    
    def check_girth(adj, n, min_girth):
        # BFS from each vertex to find shortest cycle
        # Since k=4, we can prune
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and dist[u] + 1 > dist[v]:
                        # Found a cycle
                        # Length is dist[u] + 1 + dist[v] - 2 * LCA? 
                        # Actually, in BFS, if we see a visited node (not parent), 
                        # the cycle length is dist[u] + 1 + dist[v] - 2 * dist[lca].
                        # But for simple check, we can just track depth.
                        # A simpler check: 
                        # If we encounter a visited node, we found a cycle.
                        # The length is at most dist[u] + dist[v] + 1.
                        # To be exact, we need LCA.
                        # But we just need to know if girth < min_girth.
                        # If dist[u] + dist[v] + 1 < min_girth, it's a fail.
                        # If it's larger, it might still be a shorter cycle via another path?
                        # No, BFS finds shortest path.
                        # The cycle length formed by u-v edge is dist[u] + 1 + dist[v] - 2 * dist[lca].
                        # This is hard to compute on the fly without LCA.
                        # Alternative: Use BFS to find the shortest cycle containing 'start'.
                        # But we need global girth.
                        # Standard algorithm: For each edge (u,v), remove it, find shortest path u->v.
                        # Cycle length = path_len + 1.
                        # This is O(E * (V+E)).
                        # For n=320, E=640. 640 * 960 ~ 600k ops. Fast.
                        pass
        
        # Let's use the edge-removal method for accuracy
        for u in range(n):
            for v in adj[u]:
                if u > v: continue
                # Remove edge u-v
                # Find shortest path from u to v in G - (u,v)
                dist = [-1] * n
                dist[u] = 0
                q = deque([u])
                found = False
                while q:
                    node = q.popleft()
                    if node == v:
                        cycle_len = dist[node] + 1
                        if cycle_len < min_girth:
                            return False
                        found = True
                        break
                    for w in adj[node]:
                        if w == v and node == u: continue
                        if w == u and node == v: continue
                        if dist[w] == -1:
                            dist[w] = dist[node] + 1
                            q.append(w)
                if found:
                    # We found the shortest cycle for this edge
                    # But we need to check all edges?
                    # No, if any edge forms a short cycle, girth is short.
                    # So if we find one short cycle, return False.
                    # If we finish all edges and no short cycle found, return True.
                    pass
        
        return True

    def generate_random_4_reg(n):
        # Configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i]*4)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, 4*n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Loop
            if v in adj[u]:
                return None # Multi-edge
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Search
    # Try n from 242 to 350
    # But 242 is the Moore bound, likely impossible.
    # Known best is 320.
    # We will try n in range(320, 241, -2) to find smaller ones first?
    # No, we want the SMALLEST n.
    # So we should try 242, 244, ...
    # But if 242-318 are impossible, we waste time.
    # Let's try a mix: 320 (high chance), then 318, 316...
    # If we find one at 320, we return.
    # If we have time, we try smaller.
    
    # Strategy:
    # 1. Try n=320 with many iterations.
    # 2. If found, return.
    # 3. Try n=318, 316...
    # 4. If time runs out, return LUW.
    
    # Actually, if we find a graph at 320, score is 320.
    # If we find one at 300, score is 300.
    # We should prioritize finding *any* valid graph first, then minimize.
    # But the search space for 320 is large.
    # Let's try n=320 first.
    
    best_n = 2048
    best_edges = None
    
    # Try n from 320 down to 242
    # But we need to be fast.
    # Let's try n=320, 318, 316, 314, 312, 310, 308, 306, 304, 302, 300
    # If we find one, we update best.
    # We can stop early if we find a very small one?
    # No, we want the minimum.
    # But we don't know if 242 is possible.
    # Let's just search 320 down to 242.
    
    # To save time, we will limit the number of trials per n.
    trials_per_n = 5000
    # Total trials: 40 * 5000 = 200,000.
    # Each trial: generation + check.
    # Generation: O(n). Check: O(n^2).
    # n=320, n^2=100k. 200k * 100k = 20 billion ops. Too slow.
    
    # We need a faster check or fewer trials.
    # Check girth using BFS from each vertex is O(n*k^d).
    # d=4 (girth 10/2 - 1).
    # 4^4 = 256.
    # n * 256 = 320 * 256 = 80k ops.
    # 200k trials * 80k = 16 billion. Still too slow.
    
    # We must reduce trials or use a smarter search.
    # Let's try 1000 trials for n=320.
    # If found, return.
    # If not, try n=318 with 1000 trials.
    # ...
    # Total 40 * 1000 = 40,000 trials.
    # 40k * 80k = 3.2 billion. Still might be slow in Python.
    
    # Let's optimize the check.
    # We only need to check if girth < 10.
    # We can check for cycles of length 3, 4, 6, 8.
    # Counting 3-cycles: O(n^3) or O(n*k^3).
    # Counting 4-cycles: O(n^2).
    
    # Let's try a different approach:
    # Use the LUW graph but try to **prune** it? No.
    
    # Let's just try n=320 with 5000 trials.
    # If we find it, great.
    # If not, we return LUW.
    # The probability of finding a (4,10) graph with n=320 in 5000 random tries?
    # The number of such graphs might be small relative to all 4-reg graphs.
    # But 320 is the known cage.
    
    # Let's try to be smarter.
    # We will try n=320.
    # If not found in 10s, we return LUW.
    # This is safe.
    
    # Actually, we can try n=320, 322, 324...
    # Larger n makes it easier to find a high girth graph.
    # But we want SMALL n.
    # So we must try small n.
    
    # Let's try n=320 with 2000 trials.
    
    for n in range(320, 241, -2):
        if time.time() > deadline - 10:
            break
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            adj = generate_random_4_reg(n)
            if adj is None:
                continue
            # Check girth
            # Simple BFS check for cycles < 10
            valid = True
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                q = deque([(start, -1)])
                while q:
                    u, p = q.popleft()
                    for v in adj[u]:
                        if v == p:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append((v, u))
                        else:
                            # Found a cycle
                            # Length is dist[u] + 1 + dist[v] - 2 * dist[lca]
                            # But we can just check if dist[u] + dist[v] + 1 < 10
                            # This is an upper bound on the cycle length?
                            # No, the cycle length is at most dist[u] + dist[v] + 1.
                            # If this upper bound is >= 10, it might still be < 10.
                            # We need exact length.
                            # But if dist[u] + dist[v] + 1 < 10, it's definitely a short cycle.
                            if dist[u] + dist[v] + 1 < 10:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Construct edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    return luw_construct()
