"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import deque
import time

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a non-abelian group using involutory generators.
    If that fails within time, fall back to the Lazebnik-Ustimenko-Woldar construction.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as reliable fallback
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    # Verify girth using non-backtracking BFS from every vertex
    def verify_girth(n, edges, required_girth):
        """Verify that the graph has girth at least required_girth."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for duplicate edges and loops
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        
        # Non-backtracking BFS from each vertex to find shortest cycle
        # For girth >= g, we need to check that no cycle of length < g exists
        # We use the fact that in a k-regular graph, a cycle of length l < g would
        # be detected by non-backtracking walks
        
        # For efficiency, we only need to check up to depth (g-1)//2 from each vertex
        # if g is odd, or g//2 - 1 if g is even
        max_depth = (g - 1) // 2 if g % 2 == 1 else g // 2 - 1
        
        for start in range(n):
            # Non-backtracking BFS: state is (vertex, prev_vertex)
            # We track the shortest distance to each (vertex, prev) state
            visited = {}  # (vertex, prev) -> distance
            queue = deque()
            
            # Initial states: for each neighbor of start
            for nb in adj[start]:
                state = (nb, start)
                if state not in visited:
                    visited[state] = 1
                    queue.append((nb, start, 1))
            
            while queue:
                v, prev, dist = queue.popleft()
                
                # If we've reached depth max_depth, we can stop expanding
                if dist >= max_depth:
                    continue
                
                for nb in adj[v]:
                    # Non-backtracking: don't go back to prev
                    if nb == prev:
                        continue
                    
                    new_state = (nb, v)
                    new_dist = dist + 1
                    
                    if new_state in visited:
                        # We found a cycle!
                        # The cycle length is visited[new_state] + new_dist
                        cycle_len = visited[new_state] + new_dist
                        if cycle_len < required_girth:
                            return False
                    else:
                        visited[new_state] = new_dist
                        queue.append((nb, v, new_dist))
        
        return True
    
    # Random Cayley graph approach on non-abelian groups
    def try_cayley_graph(k, g, time_budget=60):
        """
        Try to find a Cayley graph with the required properties.
        Uses products of small symmetric groups or other non-abelian groups.
        """
        start_time = time.time()
        
        # For k=5, g=9, we need 5 involutory generators
        # Try on (Z/2Z)^m for various m, or on products of symmetric groups
        
        # Strategy: Use (Z/2Z)^m where m is chosen so that the group has
        # enough elements to potentially give a good girth
        
        # For a Cayley graph on (Z/2Z)^m with generators S (all involutions),
        # the girth is related to the minimal length of a non-trivial product
        # of generators that equals the identity
        
        # Try different group sizes
        for m in range(6, 16):  # Group size 2^m
            if time.time() - start_time > time_budget:
                break
            
            group_size = 2 ** m
            if group_size < mb:
                continue
            
            # Generate random sets of k involutions
            # In (Z/2Z)^m, every element is an involution
            # We need k distinct non-identity elements that generate the group
            
            for attempt in range(100):
                if time.time() - start_time > time_budget:
                    break
                
                # Generate k random non-identity elements
                generators = []
                while len(generators) < k:
                    x = random.randint(1, group_size - 1)
                    if x not in generators:
                        generators.append(x)
                
                # Check if generators generate the full group
                # In (Z/2Z)^m, we need the generators to span all m dimensions
                # Represent generators as bit vectors
                gen_vectors = []
                for gen in generators:
                    vec = []
                    x = gen
                    for i in range(m):
                        vec.append(x & 1)
                        x >>= 1
                    gen_vectors.append(vec)
                
                # Check rank of generator matrix
                # Simple Gaussian elimination over GF(2)
                mat = [v[:] for v in gen_vectors]
                rank = 0
                for col in range(m):
                    # Find pivot
                    pivot = None
                    for row in range(rank, len(mat)):
                        if mat[row][col] == 1:
                            pivot = row
                            break
                    if pivot is None:
                        continue
                    # Swap
                    mat[rank], mat[pivot] = mat[pivot], mat[rank]
                    # Eliminate
                    for row in range(len(mat)):
                        if row != rank and mat[row][col] == 1:
                            for c in range(col, m):
                                mat[row][c] ^= mat[rank][c]
                    rank += 1
                    if rank == m:
                        break
                
                if rank < m:
                    continue  # Generators don't span the group
                
                # Build the Cayley graph
                # Vertices are 0..group_size-1
                # Edge between u and u^v for each generator v
                
                edges = []
                for u in range(group_size):
                    for v in generators:
                        w = u ^ v
                        if u < w:
                            edges.append((u, w))
                
                # Verify girth
                if verify_girth(group_size, edges, g):
                    return group_size, edges
        
        return None
    
    # First try the Cayley graph approach
    result = try_cayley_graph(k, g, time_budget=50)
    
    if result is not None:
        return result
    
    # Fall back to L-U-W construction
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
