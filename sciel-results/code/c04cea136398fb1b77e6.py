import random
import time
import numpy as np
from collections import deque

def _check_girth_bipartite(adj_u, adj_w, g):
    """
    Check if bipartite graph has girth >= g.
    adj_u: list of lists, adj_u[i] are neighbors in W for U[i].
    adj_w: list of lists, adj_w[j] are neighbors in U for W[j].
    g: minimum girth (must be even).
    """
    n_u = len(adj_u)
    n_w = len(adj_w)
    # For bipartite graph, girth is even.
    # We need to ensure no cycle of length < g.
    # Since g=10, we check for cycles of length 4, 6, 8.
    
    # Check for 4-cycles:
    # A 4-cycle exists if there are u1, u2 in U and w1, w2 in W such that
    # w1, w2 are both neighbors of u1 and u2.
    # This is equivalent to: for any pair of vertices in U, they share at most 1 neighbor in W.
    # And for any pair of vertices in W, they share at most 1 neighbor in U.
    
    # Efficient check for 4-cycles using sets:
    # For each u in U, sort its neighbors.
    # For each pair of neighbors (w1, w2) of u, check if any other u' has both w1 and w2.
    # This is O(n_u * k^2 * n_u) which is too slow.
    
    # Better: Use bitsets for neighbors.
    # adj_u_bit[u] is a bitset of W.
    # 4-cycle exists if (adj_u_bit[u1] & adj_u_bit[u2]).count() > 1.
    
    adj_u_bit = [0] * n_u
    adj_w_bit = [0] * n_w
    
    for i in range(n_u):
        mask = 0
        for j in adj_u[i]:
            mask |= (1 << j)
        adj_u_bit[i] = mask
        
    for j in range(n_w):
        mask = 0
        for i in adj_w[j]:
            mask |= (1 << i)
        adj_w_bit[j] = mask
            
    # Check 4-cycles
    for i in range(n_u):
        for j in range(i+1, n_u):
            common = adj_u_bit[i] & adj_u_bit[j]
            # Count bits
            if bin(common).count('1') > 1:
                return False
                
    for j in range(n_w):
        for i in range(j+1, n_w):
            common = adj_w_bit[i] & adj_w_bit[j]
            if bin(common).count('1') > 1:
                return False
                
    # Check 6-cycles:
    # A 6-cycle is u1-w1-u2-w2-u3-w3-u1.
    # This implies u1 and u2 share w1, u2 and u3 share w2, u3 and u1 share w3.
    # This is harder to check with just bitsets.
    # We can use BFS from each vertex up to depth 3.
    
    # For girth 10, we need to check up to depth 4 (cycles of length 8) and 5 (cycles of length 10).
    # Since g=10, we need to ensure no cycle of length 4, 6, 8.
    
    # BFS from each vertex in U up to depth 4.
    # If we reach a vertex in U at depth 4 that is already visited at depth < 4, we have a cycle.
    # Actually, in bipartite graphs, we only return to U at even depths.
    # Depth 2: W. Depth 4: U.
    # If we reach a U vertex at depth 4 that was visited at depth 0 (start) or 2?
    # Start at u0 (depth 0).
    # Neighbors in W at depth 1.
    # Neighbors in U at depth 2.
    # Neighbors in W at depth 3.
    # Neighbors in U at depth 4.
    # If any u at depth 4 is u0, we have an 8-cycle? No, u0-w1-u1-w2-u2-w3-u3-w4-u0 is length 8.
    # If u at depth 4 is u1 (visited at depth 2), we have a 6-cycle? u1-w1-u0-w2-u2-w3-u1? No.
    
    # Let's do a standard BFS for each vertex up to depth 4.
    for start in range(n_u):
        dist = [-1] * n_u
        dist[start] = 0
        queue = deque([(start, 0)])
        while queue:
            u, d = queue.popleft()
            if d == 4:
                continue
            for w in adj_u[u]:
                if dist[w] == -1: # This is for W, we need separate dist for W?
                    # We need to track distance for both U and W.
                    pass
        
    # Simpler: Just run BFS on the full graph (320 nodes) up to depth 4.
    # n=320 is small.
    adj_full = [[] for _ in range(n_u + n_w)]
    for i in range(n_u):
        for j in adj_u[i]:
            adj_full[i].append(n_u + j)
            adj_full[n_u + j].append(i)
            
    for start in range(n_u + n_w):
        dist = [-1] * (n_u + n_w)
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] == 4:
                continue
            for v in adj_full[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle?
                    # Cycle length = dist[u] + dist[v] + 1
                    # But we need to ensure it's a simple cycle.
                    # In BFS, the first time we see a visited node, it's the shortest path.
                    # If dist[u] + dist[v] + 1 < g, then girth < g.
                    # However, this detects *some* cycle, not necessarily the shortest.
                    # But if any cycle < g exists, we return False.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback to a simple random generator for other cases, though task is specific
        return _fallback_construct(k, g)
        
    start_time = time.time()
    n_half = 160
    n = 320
    
    # Bipartite random generation
    for attempt in range(500):
        if time.time() - start_time > 110:
            break
            
        # Generate random bipartite 4-regular graph
        adj_u = [[] for _ in range(n_half)]
        adj_w = [[] for _ in range(n_half)]
        
        # Use configuration model for bipartite
        stubs_u = [i for i in range(n_half) for _ in range(4)]
        stubs_w = [j for j in range(n_half) for _ in range(4)]
        random.shuffle(stubs_u)
        random.shuffle(stubs_w)
        
        valid = True
        seen_edges = set()
        
        for i in range(n_half * 4):
            u = stubs_u[i]
            w = stubs_w[i]
            if (u, w) in seen_edges:
                valid = False
                break
            seen_edges.add((u, w))
            adj_u[u].append(w)
            adj_w[w].append(u)
            
        if not valid:
            continue
            
        # Check degrees
        if any(len(a) != 4 for a in adj_u) or any(len(a) != 4 for a in adj_w):
            continue
            
        # Check girth
        if _check_girth_bipartite(adj_u, adj_w, g):
            edges = []
            for i in range(n_half):
                for j in adj_u[i]:
                    edges.append((i, n_half + j))
            return n, edges
            
    # Fallback
    return _fallback_construct(k, g)

def _fallback_construct(k, g):
    # Simple random non-bipartite
    n = 320
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    seen = set()
    for i in range(0, n*k, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v: continue
        if (min(u,v), max(u,v)) in seen: continue
        seen.add((min(u,v), max(u,v)))
        adj[u].append(v)
        adj[v].append(u)
        
    # Verify
    if all(len(a)==k for a in adj):
        # Quick girth check
        for start in range(n):
            dist = [-1]*n
            dist[start]=0
            q=deque([start])
            while q:
                u=q.popleft()
                if dist[u] >= 4: continue
                for v in adj[u]:
                    if dist[v]==-1:
                        dist[v]=dist[u]+1
                        q.append(v)
                    else:
                        if dist[u]+dist[v]+1 < g:
                            return _luw_construct(k,g)
        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Keep the old LUW as ultimate fallback
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
