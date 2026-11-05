"""Program: construct k-regular graph of girth g using random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Generate random Cayley graphs on non-abelian groups (products of symmetric groups)
    using k involutory generators, then verify girth via BFS.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    deadline = time.time() + 115  # Leave 5s margin
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # Verify girth using non-backtracking walks from every vertex
    def verify_girth(n, edges, g):
        """Check that no cycle shorter than g exists using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS tracking (node, prev) to avoid backtracking
        # A cycle of length L exists if we can reach a neighbor of start via a path of length L-1
        # without backtracking
        for start in range(n):
            # BFS state: (current, parent) -> distance
            # We want to find if any cycle of length < g exists
            visited = {}  # (node, parent) -> dist
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
            
            while queue:
                node, parent, dist = queue.popleft()
                if dist >= g:
                    continue
                if dist == g - 1 and parent == start:
                    return False  # Found cycle of length g-1 < g
                if dist == g - 1:
                    # Check if node is adjacent to start (would form cycle of length g)
                    # Actually we need cycle < g, so check if node == start's neighbor
                    pass
                for nb in adj[node]:
                    if nb == parent:
                        continue  # Don't backtrack
                    if dist + 1 >= g:
                        continue
                    key = (nb, node)
                    if key not in visited:
                        visited[key] = dist + 1
                        queue.append((nb, node, dist + 1))
                    # If nb is start and dist+1 < g, we found a short cycle
                    if nb == start and dist + 1 < g:
                        return False
        return True
    
    # Better girth verification: BFS from each vertex, check if we can return to start
    # with path length < g (non-backtracking)
    def verify_girth_v2(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # State: (current_node, previous_node) -> min distance
            dist = {}
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                dist[(nb, start)] = 1
            
            while queue:
                node, prev, d = queue.popleft()
                if d >= g:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    nd = d + 1
                    if nd >= g:
                        continue
                    if nb == start:
                        # Found cycle of length nd < g
                        return False
                    key = (nb, node)
                    if key not in dist or dist[key] > nd:
                        dist[key] = nd
                        queue.append((nb, node, nd))
        return True
    
    # Generate a random Cayley graph on a group
    # We'll use the group S_m x S_m or similar products
    # For k=5, we need 5 involutions
    
    def try_cayley_graph(k, g, max_attempts=50):
        """Try to find a valid Cayley graph with the given parameters."""
        # For k=5, g=9, we need a group with 5 involutory generators
        # such that the resulting Cayley graph has girth >= 9
        
        # Try different group sizes
        # The group should be large enough that random generators don't create short cycles
        # but small enough that we can enumerate it
        
        # Strategy: Use (Z/2Z)^m for small m, or S_m for small m
        # Actually, for non-abelian, let's try S_m
        
        # For S_m, involutions are products of disjoint transpositions
        # We need k=5 such involutions that generate the group (or a subgroup)
        
        # Let's try S_6 or S_7
        for m in [6, 7, 8]:
            if time.time() > deadline:
                break
            
            # Generate all elements of S_m as tuples
            # This is m! elements, which is 720 for S_6, 5040 for S_7
            import itertools
            
            if m > 7:
                # Too many elements, skip
                continue
            
            elements = list(itertools.permutations(range(m)))
            n = len(elements)
            elem_to_idx = {e: i for i, e in enumerate(elements)}
            
            # Multiplication table
            def perm_mul(p, q):
                # (p * q)(i) = p(q(i))
                return tuple(p[q[i]] for i in range(m))
            
            # Precompute multiplication
            mul_table = {}
            for p in elements:
                for q in elements:
                    mul_table[(p, q)] = elem_to_idx[perm_mul(p, q)]
            
            identity = tuple(range(m))
            
            # Generate random involutions
            # An involution in S_m is a product of disjoint transpositions
            def random_involution(m):
                # Randomly choose a set of disjoint transpositions
                remaining = set(range(m))
                result = list(range(m))
                while len(remaining) >= 2:
                    if random.random() < 0.5:
                        # Add a transposition
                        a, b = random.sample(list(remaining), 2)
                        result[a] = b
                        result[b] = a
                        remaining.discard(a)
                        remaining.discard(b)
                    else:
                        break
                return tuple(result)
            
            # Try multiple random sets of k involutions
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                # Generate k random involutions
                gens = []
                for _ in range(k):
                    inv = random_involution(m)
                    # Ensure it's not identity
                    if inv == identity:
                        continue
                    # Ensure it's actually an involution (it should be by construction)
                    # Check: inv * inv == identity
                    if perm_mul(inv, inv) != identity:
                        continue
                    # Avoid duplicates
                    if inv in gens:
                        continue
                    gens.append(inv)
                
                if len(gens) < k:
                    continue
                
                # Build the Cayley graph
                # Vertex i is connected to vertex j if j = i * g for some generator g
                # Since generators are involutions, the graph is undirected
                edges = set()
                adj = [[] for _ in range(n)]
                
                for i, elem in enumerate(elements):
                    for gen in gens:
                        j = mul_table[(elem, gen)]
                        if i < j:
                            edges.add((i, j))
                        adj[i].append(j)
                        adj[j].append(i)
                
                # Check degree
                for i in range(n):
                    if len(adj[i]) != k:
                        break
                else:
                    # All vertices have degree k
                    # Verify girth
                    if verify_girth_v2(n, list(edges), g):
                        return n, list(edges)
        
        return None
    
    # Fallback: L-U-W construction
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
    
    # Try Cayley graph approach first
    result = try_cayley_graph(k, g)
    if result is not None:
        return result
    
    # Fall back to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
