import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 108  # Safety margin
    
    # LUW Construction as fallback
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
    
    # Voltage graph / Cayley graph approach
    # For a k-regular Cayley graph on Z_n with generators S = {s1, ..., sk} where S is symmetric
    # (i.e., -s_i is also in S for each i, or we pair generators), the girth is determined
    # by the shortest non-backtracking walk that returns to 0.
    
    def cayley_girth(n, generators):
        """
        Compute the girth of a Cayley graph on Z_n with given generators.
        generators: list of k integers (mod n), each representing an edge direction.
        The graph is k-regular if we consider each generator and its inverse as separate edges,
        OR if generators come in pairs {s, -s} and we have k/2 such pairs (k must be even).
        
        For odd k (like k=5), we need a different approach. Let's use a bipartite-like structure
        or a different group.
        
        Actually, for k=5 (odd), a Cayley graph on an abelian group won't work directly because
        the degree must be even for a symmetric generating set. Let's use a different approach.
        
        Alternative: Use a graph that is a covering of a smaller graph, or use a non-abelian group.
        Or, we can construct a 5-regular graph by taking a 4-regular Cayley graph and adding
        a perfect matching, but that's complex.
        
        Let me think differently. For k=5, g=9, the Moore bound is:
        n >= 1 + k*(1 + (k-1) + ... + (k-1)^(g/2 - 1)) for even g
        g=9 is odd, so we use the odd girth bound:
        n >= 2*(k-1)^((g-1)/2) = 2*4^4 = 512
        
        The best known is 1116. LUW gives 6250.
        
        Let me try a different approach: use a circulant graph. A circulant graph C_n(S) has
        vertices 0..n-1 and edges (i, i+s mod n) for s in S. For it to be k-regular and simple,
        we need |S| = k and S symmetric (if -s in S, then we count it once).
        
        For k=5 (odd), we can't have a symmetric S of size 5 in Z_n (since s and -s come in pairs).
        Unless we use a directed graph, but we need undirected.
        
        So for odd k, we need a different construction. Let's use the fact that we can take
        a 5-regular graph by combining two parts, or use a non-circulant approach.
        
        Actually, let me try a random 5-regular graph approach with girth checking.
        """
        # For now, return -1 to indicate this approach needs more work
        return -1
    
    def random_5_regular(n):
        """Generate a random 5-regular graph on n vertices."""
        # Each vertex needs degree 5
        # Use configuration model
        half_edges = [i for i in range(n) for _ in range(5)]
        random.shuffle(half_edges)
        
        adj = [set() for _ in range(n)]
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                # Self-loop, retry
                continue
            if v in adj[u]:
                # Multi-edge, retry
                continue
            adj[u].add(v)
            adj[v].add(u)
        
        # Check if all degrees are exactly 5
        for i in range(n):
            if len(adj[i]) != 5:
                return None
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, edges
    
    def check_girth_bfs(n, edges, target_girth):
        """
        Check if the graph has no cycle shorter than target_girth.
        Uses BFS from each vertex to find shortest cycle.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS to find shortest cycle
        # A cycle of length L starting at v means there's a path of length L-1 from a neighbor of v
        # back to v.
        
        # More efficient: for each edge (u,v), check if there's a path of length < target_girth-1
        # from u to v not using edge (u,v). If yes, cycle of length < target_girth.
        
        # Even more efficient: use non-backtracking BFS from each vertex up to depth (target_girth-1)//2
        # If we return to start, we found a cycle.
        
        half_depth = (target_girth - 1) // 2
        
        for start in range(n):
            # BFS with non-backtracking constraint
            # State: (current_vertex, previous_vertex)
            visited = set()
            queue = deque()
            
            for nb in adj[start]:
                queue.append((start, nb, 1))
                visited.add((start, nb))
            
            while queue:
                if time.time() > deadline - 5:
                    return False
                prev, curr, dist = queue.popleft()
                
                if dist >= half_depth:
                    continue
                
                for nb in adj[curr]:
                    if nb == prev:
                        continue  # Non-backtracking
                    if nb == start:
                        # Found a cycle of length dist+1
                        if dist + 1 < target_girth:
                            return False
                        continue
                    if (curr, nb) not in visited:
                        visited.add((curr, nb))
                        queue.append((curr, nb, dist + 1))
        
        return True
    
    # Try to find a smaller graph using random 5-regular graphs
    best_n = None
    best_edges = None
    
    # Start with a reasonable upper bound and try to find smaller graphs
    # The Moore bound for k=5, g=9 is 512, best known is 1116
    # Try a range of sizes
    
    if k == 5 and g == 9:
        # Try several random graphs with different sizes
        sizes_to_try = [600, 700, 800, 900, 1000, 1116]
        
        for n_try in sizes_to_try:
            if time.time() > deadline - 10:
                break
            
            # Try multiple random graphs for this size
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                result = random_5_regular(n_try)
                if result is None:
                    continue
                
                n, edges = result
                if check_girth_bfs(n, edges, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    break  # Found one for this size, try smaller
    
    # If we found a valid graph, return it
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    n, edges = luw_construct(k, g)
    return n, edges
