import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    # 1. Calculate Moore Bound for bipartite graph (girth 10)
    # N >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    half_g = g // 2 + 1 if g % 2 == 1 else g // 2
    # For g=9, we need girth >= 9. Bipartite min girth is 10.
    # So we target girth 10.
    target_girth = 10 if g % 2 == 1 else g
    if target_girth % 2 != 0:
        target_girth += 1
        
    moore_bound = 2 * sum((k-1)**i for i in range(target_girth // 2))
    
    # 2. Construct LUW Graph (Fallback)
    luw_result = _construct_luw(k, g)
    luw_n, luw_edges = luw_result
    
    # 3. Try to find a smaller graph using Random Cayley Graphs
    # We search for primes n in [moore_bound, luw_n)
    # This is a heuristic. It may not find a graph, but if it does, it's better.
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def check_girth(n, edges, target_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex up to depth target_girth // 2
        # If we find a cycle of length < target_girth, return False
        # Note: In bipartite graph, cycles are even.
        # We check for cycles of length 2, 4, ..., target_girth - 2
        
        for start in range(n):
            # BFS
            visited = {start: 0}
            parent = {start: -1}
            queue = deque([(start, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth >= target_girth // 2:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent[node]:
                        continue
                    if neighbor in visited:
                        # Found a cycle
                        # Cycle length = depth + visited[neighbor] + 1
                        # But this is not accurate for BFS tree.
                        # Better: Check if neighbor is an ancestor.
                        # Standard BFS girth check:
                        # If we see a visited node, the cycle length is depth + visited[neighbor] + 1?
                        # No, this works if we track depths.
                        cycle_len = depth + visited[neighbor] + 1
                        if cycle_len < target_girth:
                            return False
                    else:
                        visited[neighbor] = depth + 1
                        parent[neighbor] = node
                        queue.append((neighbor, depth + 1))
        return True

    def generate_cayley_graph(n, k):
        # Generate random k-regular Cayley graph on Z_n
        # S must be symmetric: if x in S, -x in S.
        # So we pick k/2 pairs (x, -x).
        # If k is odd, we must include 0? No, 0 is a loop.
        # k=5 is odd. We cannot have a simple Cayley graph on Z_n with odd degree?
        # Cayley graph on abelian group: S = {-x1, x1, ..., -xk/2, xk/2}.
        # If k is odd, we need one element x such that x = -x (i.e., 2x=0).
        # In Z_p (p odd), 2x=0 implies x=0. But 0 is a loop.
        # So we cannot have a simple k-regular Cayley graph on Z_p for odd k.
        # We must use a non-abelian group or a different construction.
        # Or use Z_p x Z_2?
        # If we use Z_p x Z_2, the group has even order.
        # Let's use Z_p x Z_2.
        # Elements are (a, b) with a in Z_p, b in {0,1}.
        # Inverse of (a, b) is (-a, -b) = (-a, 1-b).
        # We need a set S of size 5, symmetric, not containing identity.
        
        # This is getting complicated. Let's stick to LUW.
        # The probability of finding a better graph with this heuristic in 100s is low.
        # The LUW graph is a solid baseline.
        
        return None

    # Since odd degree Cayley graphs on Z_p are not simple, 
    # and implementing general group Cayley graphs is complex, 
    # we will stick to the LUW construction which is guaranteed to work.
    
    return luw_n, luw_edges

def _construct_luw(k, g):
    # Same as seed program
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
