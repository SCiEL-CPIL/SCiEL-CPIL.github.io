import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore bound for k=3, g=16 is 510.
    # N must be even.
    # Known best is 936.
    # We will try N from 512 upwards.
    
    # Helper: Generate a random 3-regular graph
    def generate_random_3_regular(n):
        if n < 4 or n % 2 != 0:
            return None
        # Use configuration model
        # List of stubs: each vertex appears 3 times
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        edges = set()
        # Pair up stubs
        for i in range(0, n*3, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, retry or fix
                # For simplicity, restart if self-loop occurs (rare for large n)
                # But for small n, might happen.
                return None
            if (v, u) in edges:
                # Multi-edge, restart
                return None
            edges.add((u, v))
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Verify degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return adj

    # Helper: Check girth >= g
    # BFS from each vertex up to depth g//2 - 1
    # If we find a cycle of length < g, return False
    def has_girth_at_least(adj, n, g):
        half_g = g // 2
        # We need to check cycles of length < g.
        # A cycle of length L corresponds to a path of length L//2 from a vertex to itself (if L even)
        # or a path from u to v where dist(u,v) = L//2 and there is an edge? No.
        # Standard method: BFS from each vertex, tracking parents.
        # If we reach a visited node that is not the parent, we found a cycle.
        # Cycle length = depth[current] + depth[visited] + 1 (if edge exists)
        # Actually, simpler:
        # For each vertex s, BFS up to depth half_g - 1.
        # If we encounter a node v that was visited in a previous BFS? No, that's not right.
        
        # Correct BFS Girth Check:
        # For each vertex s:
        #   BFS up to depth half_g - 1.
        #   If we find a cycle, its length is 2 * depth + 1 (if odd) or 2 * depth (if even)?
        #   Actually, if we find a neighbor of s at depth d, cycle is d+1.
        #   If we find a neighbor of a node at depth d at depth d+1, and that node is already visited at depth d'...
        
        # Efficient check for girth >= g:
        # No cycle of length < g.
        # This is equivalent to: for any edge (u,v), the distance between u and v in G \ {(u,v)} is >= g-1.
        # This is hard to check directly.
        
        # Let's use the standard BFS from each node.
        # For s in range(n):
        #   dist = [-1]*n
        #   parent = [-1]*n
        #   dist[s] = 0
        #   queue = deque([s])
        #   while queue:
        #     u = queue.popleft()
        #     if dist[u] >= half_g - 1: continue
        #     for v in adj[u]:
        #       if dist[v] == -1:
        #         dist[v] = dist[u] + 1
        #         parent[v] = u
        #         queue.append(v)
        #       else:
        #         # Found a cycle
        #         # If v is parent of u, ignore
        #         if v == parent[u]: continue
        #         # Cycle length = dist[u] + dist[v] + 1
        #         # But this might overcount if the path to v is not disjoint from path to u.
        #         # Actually, the cycle formed by path s->u, path s->v, and edge (u,v)
        #         # has length dist[u] + dist[v] + 1.
        #         # However, the shortest cycle through s might be shorter.
        #         # If dist[u] + dist[v] + 1 < g, return False.
        #         if dist[u] + dist[v] + 1 < g:
        #             return False
        #   # End while
        # End for
        # This works!
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    best_n = None
    best_edges = None
    
    # We know 936 is possible. Let's try to find something close.
    # Start from 512, step by 2.
    # To save time, we can jump. But let's try sequentially.
    
    # Optimization: Use a fixed seed for reproducibility? No, random is fine.
    # We need to be careful with time.
    
    # Let's try a range of N.
    # If N is too small, probability is 0.
    # For N=936, probability is high.
    # Let's start from 800 to 1000 to be safe, then go down if time permits?
    # Or start from 512 and go up.
    
    # Given the 120s limit, and BFS cost ~ O(N * 3^(g/2)), 
    # For N=1000, g=16, cost is ~ 1000 * 2187 = 2.1M operations.
    # Python can do ~10M ops/sec. So ~0.2s per graph.
    # We can try ~500 graphs in 100s.
    
    # Let's start from N=512 and go up.
    # But if N is too small, we waste time.
    # Heuristic: Start from N where 2*log2(N) >= g.
    # 2*log2(N) >= 16 => log2(N) >= 8 => N >= 256.
    # But girth is probabilistic.
    # Let's try N from 512 to 1000.
    
    current_n = 512
    while current_n <= 1000:
        if time.time() > deadline:
            break
            
        # Try a few random graphs of this size
        # If we find one, return it.
        # If not, increase N.
        
        # How many trials per N?
        # If N is small, probability is low. 
        # If N is large, probability is high.
        # Let's do 10 trials for small N, 5 for large N?
        # Actually, just keep trying until time runs out.
        
        # Better: Try to find a valid graph for current_n.
        # If found, return.
        # If not, increment current_n by 2.
        
        # To avoid spending all time on N=512 (which is likely impossible),
        # we can set a max trials per N.
        max_trials_per_n = 50
        
        found = False
        for _ in range(max_trials_per_n):
            if time.time() > deadline:
                break
                
            adj = generate_random_3_regular(current_n)
            if adj is None:
                continue
                
            if has_girth_at_least(adj, current_n, g):
                # Convert adj to edge list
                edges = []
                for u in range(current_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return current_n, edges
                
        # If not found, try larger N
        current_n += 2

    # If we didn't find one, return the last generated graph (even if girth < g) 
    # or a known small graph?
    # The task says "invalid result ... scores 0".
    # So we MUST return a valid graph.
    # If we run out of time, we should return the best candidate we had?
    # But if it's invalid, score is 0.
    # We should ensure we return *something* valid if possible.
    # A 3-regular graph with girth < 16 is valid as a graph, but fails the girth constraint.
    # The contract says: "no cycle may be shorter than g".
    # So if we can't find one, we fail.
    
    # Fallback: Return the LUW graph from the seed?
    # The seed graph is valid (girth >= 16? Seed says "girth is at least d+5". 
    # For g=16, d=11? 
    # Seed: d = max(3, g-5 if (g-5)%2 else g-4).
    # g=16 => g-5=11 (odd) => d=11.
    # Girth >= 11+5 = 16.
    # So the seed graph is VALID.
    # We should fallback to the seed construction if our search fails.
    
    # Re-implement seed logic as fallback
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
