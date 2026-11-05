import time
import random
from itertools import product, combinations_with_replacement

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Try to find a small Circulant Graph
    # Moore bound for k=5, g=9 is 426.
    # We search n from 426 upwards.
    # For each n, we try random circulant sets.
    
    # If k is odd, one generator must be n/2 (if n even) or we use 3 generators?
    # Degree 5.
    # If n is even, S = {a, -a, b, -b, n/2}.
    # If n is odd, S = {a, -a, b, -b, c, -c}? No, that's degree 6.
    # For odd degree k, if n is odd, we can't have a self-inverse element (n/2).
    # So we need 2.5 pairs? Impossible.
    # Wait, if n is odd, all elements have order > 2.
    # So generators come in pairs {s, -s}.
    # Degree must be even?
    # No, a circulant graph on odd n can have odd degree?
    # If S is symmetric, |S| must be even?
    # If s in S, -s in S.
    # If s = -s mod n, then 2s = 0 mod n. If n is odd, s=0.
    # So for odd n, S must be a union of pairs {s, -s}.
    # So degree must be even.
    # But k=5 is odd.
    # So n MUST be even for a circulant graph of odd degree.
    
    for n in range(426, 1000, 2): # n must be even
        if time.time() > deadline:
            break
            
        # Try to find S = {a, -a, b, -b, n/2}
        # a, b in 1..n/2-1, a != b
        # Check girth conditions
        
        # We can sample random a, b
        found = False
        for _ in range(50): # Try 50 random pairs per n
            if time.time() > deadline:
                break
            a = random.randint(1, n//2 - 1)
            b = random.randint(1, n//2 - 1)
            if a == b:
                continue
            
            S = [a, n-a, b, n-b, n//2]
            
            # Check for short cycles
            # A cycle of length L corresponds to a sum of L elements from S (with signs) equal to 0 mod n.
            # Since S is symmetric, we just need to check if there exist s1...sL in S such that sum(si) = 0.
            # For L=3: s1+s2+s3=0.
            # For L=4: s1+s2+s3+s4=0.
            # ...
            
            valid = True
            # Check 3-cycles
            # s1+s2+s3=0.
            # Since S contains -s for every s, this is equivalent to s1+s2 = -s3 = s3' (if s3' in S).
            # So we check if any sum of 2 elements is in S.
            sums_2 = set()
            for s1 in S:
                for s2 in S:
                    sums_2.add((s1+s2) % n)
            if S[0] in sums_2: # Check if any s in S is a sum of 2
                # Actually, we need s1+s2+s3=0.
                # This means s1+s2 = -s3.
                # Since -s3 is in S, we check if any sum of 2 elements is in S.
                valid = False
                break
            
            if not valid:
                continue
                
            # Check 4-cycles
            # s1+s2+s3+s4=0.
            # s1+s2 = -(s3+s4).
            # So we check if any sum of 2 elements is the negative of another sum of 2 elements.
            # i.e., is there x in Sums2 such that -x is also in Sums2?
            # Note: x = s1+s2. -x = -s1-s2.
            # If x != 0, and -x in Sums2, then we have a 4-cycle.
            # But we must ensure the sequence is a valid cycle (distinct vertices).
            # For circulant graphs, if sum of steps is 0, it's a closed walk.
            # It might backtrack.
            # Backtracking: s, -s, s, -s. Sum 0. Length 4.
            # This is a 2-cycle traversed twice? No, in a simple graph, we can't traverse the same edge back and forth in a "simple cycle".
            # The definition of girth uses simple cycles.
            # A closed walk of length 4 that backtracks is not a simple cycle.
            # So we must exclude backtracking walks.
            
            # This check is getting complex.
            # Let's just build the graph and BFS for girth.
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    if v > u:
                        adj[u].append(v)
                        adj[v].append(u)
                    # Note: if s = n/2, v = u + n/2.
                    # We add edge (u, v) only once.
            
            # Check degree
            # It should be 5.
            
            # Check girth
            # BFS from every vertex? Too slow for n=1000.
            # BFS from a few vertices?
            # Or use the property that girth is the shortest cycle.
            
            # We can check for cycles of length < g using a specific algorithm.
            # For small n, BFS from each vertex is O(n^2). 1000^2 = 1e6. Fast.
            
            girth = n + 1
            for i in range(n):
                # BFS from i
                dist = [-1] * n
                parent = [-1] * n
                dist[i] = 0
                queue = [i]
                found_cycle = False
                while queue and not found_cycle:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]: # Not backtracking
                            # Found a cycle
                            # Length = dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
                            if girth < g:
                                found_cycle = True
                                break
                if girth < g:
                    break
            
            if girth >= g:
                # Found a valid graph
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback to LUW construction
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
