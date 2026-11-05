import time
from collections import deque
from math import gcd
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Moore bound for k=4, g=10 is 242.
    # We start from 243 to be safe, or 242.
    # The prompt says best lower bound is 243.
    n_start = 243
    
    # We will search for a circulant graph C_n({+/- a, +/- b})
    # k must be 4. If k != 4, this won't work directly, but the task says "evaluator passes this task's degree k".
    # The task is specifically for k=4, g=10.
    # If k is not 4, we might need a fallback, but the prompt implies k=4.
    
    if k != 4:
        # Fallback to LUW or simple construction if k is different
        # But for this specific task, k=4.
        pass

    for n in range(n_start, 500):  # Upper bound on n to search
        if time.time() > deadline:
            break
            
        # For each n, try pairs (a, b)
        # a, b are steps.
        # 1 <= a < b < n/2
        # Also a != b, a != n-b, etc.
        # We just iterate a and b.
        
        max_step = n // 2
        for a in range(1, max_step):
            if time.time() > deadline:
                break
            for b in range(a + 1, max_step):
                # Check connectivity: gcd(n, a, b) == 1
                if gcd(gcd(n, a), b) != 1:
                    continue
                
                # Check girth using BFS
                # Steps: a, -a, b, -b
                steps = [a, -a, b, -b]
                
                # We can prune if 2a == 0 or 2b == 0 (creates 2-cycle? No, simple graph, but 4-cycle 0-a-0-a?)
                # In a simple graph, 0 and a are connected.
                # Cycle 0-a-0 is length 2? No, edges are undirected.
                # The walk 0->a->0 is length 2, but it's not a simple cycle.
                # Simple cycle requires distinct vertices.
                # So we just look for simple cycles.
                
                # BFS from 0
                dist = [-1] * n
                parent = [-1] * n
                dist[0] = 0
                q = deque([0])
                min_cycle = 10**9
                valid = True
                
                while q:
                    u = q.popleft()
                    du = dist[u]
                    if du >= 4: # We only need to check up to depth 4 to find cycles <= 8?
                        # Actually, to find cycles of length < 10, we need to look at depths summing to < 9.
                        # If dist[u] + dist[v] + 1 < 10, we have a short cycle.
                        # Max dist needed: if dist[u]=4, dist[v] can be up to 4 (4+4+1=9).
                        # So we need to process nodes with dist up to 4.
                        # But if we stop expanding at 4, we might miss cycles where one side is 5?
                        # If dist[u]=5, dist[v] must be 0? No.
                        # If there is a cycle of length 9, it splits into 4 and 4 (via 0)?
                        # Yes, any cycle passing through 0 splits into two paths from 0.
                        # The lengths $d_1, d_2$ satisfy $d_1 + d_2 = L$.
                        # If $L < 10$, then $\min(d_1, d_2) \le 4$.
                        # So if we find a node $v$ with dist $\le 4$ connected to a node $u$ with dist $\le 4$ such that $dist[u] + dist[v] + 1 < 10$, we found a short cycle.
                        # Actually, if we process all nodes with dist $\le 4$, we will see all edges between them.
                        # So we can stop expanding nodes with dist > 4.
                        pass
                    
                    for s in steps:
                        v = (u + s) % n
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            if du + 1 <= 4: # Only enqueue if we might use it for cycle detection
                                q.append(v)
                        else:
                            # v is already visited
                            # Check if this edge (u,v) creates a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # But we must ensure it's a simple cycle?
                            # BFS tree edges don't form cycles. Non-tree edges do.
                            # If v is a neighbor of u, and v is visited, and v != parent[u] and u != parent[v]?
                            # Actually, in undirected graph, if we see an edge to a visited node,
                            # and it's not the parent, it forms a cycle.
                            # However, we might see the same edge twice (u->v and v->u).
                            # We should only count it once.
                            # Since we iterate u, and check all neighbors, we will see u->v.
                            # If v is visited, we check cycle length.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                            if min_cycle < 10:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid and min_cycle >= 10:
                    # Construct edges
                    edges = []
                    for i in range(n):
                        for s in [a, b]:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    # Note: steps are a, -a, b, -b.
                    # Edge (i, i+a) is same as (i+a, i).
                    # So we just add (i, i+a) and (i, i+b) for i < j.
                    return n, edges

    # If not found, return a default valid graph (e.g. LUW)
    # But we should try to return something valid.
    # The LUW graph from the seed is valid but large.
    # Let's just return the LUW construction if search fails.
    return _luw_construct(k, g)

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
