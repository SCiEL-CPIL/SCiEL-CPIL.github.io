import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        """
        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth using BFS from each vertex (or optimized non-backtracking)
        # For small graphs, BFS from each vertex is fine.
        # Girth is the length of the shortest cycle.
        # A cycle of length L implies a path of length L-1 from a vertex to one of its neighbors.
        # So we look for the shortest path from u to v where v is a neighbor of u, with length >= 2.
        # Actually, shortest cycle through u is the min over neighbors v of (dist(u, v) + 1) where dist is shortest path not using the edge (u,v) directly?
        # Simpler: BFS from u, but don't traverse back to u.
        # Or: BFS from all vertices, find shortest cycle.
        
        # Optimized check:
        # For each vertex u, run BFS to find the shortest cycle containing u.
        # We can do this by doing BFS from u, but when we reach a neighbor v of u, the cycle length is dist[u][v] + 1.
        # We must ensure the path doesn't use the edge (u,v) as the first step?
        # Actually, if we start BFS from u, and we visit a neighbor v at distance d, the cycle is d+1.
        # But if d=1, that's just the edge, not a cycle. So we need d >= 2.
        # However, BFS finds shortest paths. If the shortest path from u to v has length 1, that's the edge.
        # If there is a cycle of length 3, u-v-w-u, then dist(u, v) via w is 2. So cycle length is 3.
        # So we just need to find the minimum (dist(u, v) + 1) for all u, and all neighbors v of u, where dist(u, v) >= 2.
        # Note: dist(u, v) is the standard shortest path distance.
        
        # Let's just do BFS from each u.
        for u in range(n):
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= g:  # If we've already found a path of length g, no shorter cycle can exist through this level
                    break
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        q.append(v)
                    else:
                        # Found a cycle
                        # The cycle length is dist[curr] + 1 + dist[v] ? No.
                        # This is for general graph. 
                        # For cycle containing u:
                        # We are looking for a path from u to a neighbor v of u, of length L.
                        # Then cycle length is L + 1.
                        # In BFS from u, if we encounter an edge (curr, v) where v is a neighbor of u:
                        # Then we have a path u -> ... -> curr -> v -> u.
                        # Length = dist[curr] + 1 + 1 = dist[curr] + 2.
                        # Wait, if v is a neighbor of u, and we are at curr, and curr is connected to v.
                        # Path: u ... curr - v - u.
                        # Length: dist[u][curr] + 1 (curr to v) + 1 (v to u).
                        # So cycle length = dist[curr] + 2.
                        
                        # But we only care if v is a neighbor of u.
                        if v in adj[u] and v != curr: # v != curr is redundant if no loops, but safe
                            cycle_len = dist[curr] + 2
                            if cycle_len < g:
                                return False
                        # Also, if v is already visited and is not the parent, it might form a cycle not containing u.
                        # But we are checking girth, which is the minimum cycle in the whole graph.
                        # If any cycle is shorter than g, we fail.
                        # The standard BFS from a single source finds shortest cycles through that source.
                        # To find the global girth, we need to check all sources?
                        # Yes.
                        
                        # Actually, the check `if v in adj[u]` only catches cycles through u.
                        # What about a cycle not through u?
                        # It will be caught when we run BFS from a vertex on that cycle.
                        # So checking all u is correct.
                        
                        # However, there is a subtle bug. 
                        # If we find a shorter path to v, we don't update dist. 
                        # But if we find a cycle, we should check if it's shorter than g.
                        # The cycle found by edge (curr, v) where dist[v] != -1 has length dist[curr] + 1 + dist[v].
                        # This cycle might not contain u.
                        # So we should check ALL back edges.
                        
                        # Let's refine:
                        # When we see an edge (curr, v) where dist[v] != -1:
                        # We have a cycle of length dist[curr] + 1 + dist[v].
                        # If this length < g, return False.
                        
                        # BUT, this only works if the path from u to curr and u to v are disjoint except at u?
                        # No, BFS tree paths are disjoint except at u? Not necessarily.
                        # However, the length dist[curr] + 1 + dist[v] is a valid upper bound on the cycle length.
                        # If it's < g, then girth < g.
                        # So this is a valid check.
                        
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        # Original LUW construction
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # 1. Try LUW family
    q = k
    min_d = max(3, g - 5)
    
    # Sort d values to try smaller ones first? Or specific ones?
    # The seed tried min_d to min_d+10.
    # Let's try a range.
    for d_val in range(min_d, min_d + 20):
        if time.time() > deadline - 20:
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            if n < best_n:
                # Verify before accepting
                if verify_graph(n, edges, k, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            pass

    # 2. If LUW didn't find a good one (or if we want to try something else),
    # try a random regular graph approach? 
    # For k=4, g=10, N=320 is the target.
    # LUW(4, 5) might produce N > 320.
    # If so, we need a better construction.
    
    # Is there a known (4,10)-cage construction that is smaller?
    # The Robertson graph is the unique (4,10)-cage with 320 vertices.
    # It is a cubic graph? No, 4-regular.
    # It is a voltage graph on K_{4,4} ? Or something else.
    
    # Let's try to construct a random 4-regular graph with N=320 and check girth.
    # Probability of girth >= 10 is low, but maybe?
    # Actually, for large girth, random graphs are bad.
    # But we have 120s.
    
    # Alternative: Use the known structure of the Robertson graph if we can derive it.
    # The Robertson graph is a (4,10)-cage.
    # It is a quotient of a tree?
    
    # Let's stick to LUW but try to optimize the search or parameters.
    # Maybe q can be different? No, q=k.
    
    # What if we try to find a subgraph of a larger graph?
    # No, must be k-regular.
    
    # Let's try a different algebraic construction if LUW fails to beat 320.
    # But we don't know the current best_n.
    
    # If best_n is still inf or large, try random search for a specific N?
    # If we guess N=320, we can try to build a 4-regular graph on 320 vertices and check girth.
    # But finding a specific one is hard.
    
    # Let's assume LUW(4,5) is the intended solution and check its size.
    # If LUW(4,5) has size 320, we are done.
    # If it's larger, we might be stuck unless we implement the Robertson graph construction.
    
    # For now, return the best found.
    if best_edges is None:
        # Fallback: Return a trivial valid graph if possible?
        # A cycle of length g is 2-regular. If k=2, this works.
        # If k > 2, we need a k-regular graph with girth >= g.
        # The Moore bound is a lower bound on N.
        # If we can't find one, we might return an invalid graph or crash.
        # But the task implies a solution exists.
        
        # Let's try a very large d to ensure girth, even if N is huge.
        # This ensures we return *something* valid, even if score is bad.
        try:
            d_val = g + 10
            n, edges = build_luw_graph(q, d_val)
            if verify_graph(n, edges, k, g):
                best_n = n
                best_edges = edges
        except Exception:
            pass

    return best_n, best_edges
