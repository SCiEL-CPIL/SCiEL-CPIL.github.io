import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Fallback: LUW graph (current implementation)
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

    # ABS construction: Cayley graph on upper triangular unipotent matrices over GF(q)
    def abs_construct(k, g):
        # For k=5, we use q=5 (must be prime for GF(q))
        # The group is the set of upper triangular 3x3 matrices with 1s on diagonal
        # Represented as (a, b, c) where matrix is [[1, a, c], [0, 1, b], [0, 0, 1]]
        # Group operation: (a1,b1,c1) * (a2,b2,c2) = (a1+a2, b1+b2, c1+c2+a1*b2) mod q
        
        q = k  # For k=5, q=5
        if q < 3:
            return None
        
        # Precompute group elements
        # There are q^3 elements
        elements = []
        elem_index = {}
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    elem = (a, b, c)
                    elem_index[elem] = len(elements)
                    elements.append(elem)
        
        n = len(elements)
        
        # Define generators: elementary matrices
        # E1 = (1, 0, 0), E2 = (0, 1, 0), E3 = (0, 0, 1)
        # and their inverses (which are (-1, 0, 0), (0, -1, 0), (0, 0, -1))
        # For q=5, -1 = 4 mod 5
        
        # We need k=5 generators (or 5 pairs for undirected)
        # Let's use: (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1)
        # and their inverses
        
        # Actually, for a k-regular graph, we need k generators (counting inverses)
        # Let's pick 5 generators such that none is the inverse of another (to avoid 2-cycles)
        
        # Generators:
        gens = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1)
        ]
        
        # Inverses:
        def inverse(elem):
            a, b, c = elem
            return ((-a) % q, (-b) % q, (-c) % q)
        
        # Check that no generator is the inverse of another
        gen_set = set(gens)
        valid_gens = True
        for g1 in gens:
            inv_g1 = inverse(g1)
            if inv_g1 in gen_set:
                valid_gens = False
                break
        
        if not valid_gens:
            # Try different generators
            gens = [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 1),
                (2, 1, 0)
            ]
            gen_set = set(gens)
            valid_gens = True
            for g1 in gens:
                inv_g1 = inverse(g1)
                if inv_g1 in gen_set:
                    valid_gens = False
                    break
        
        if not valid_gens:
            return None
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        edges = set()
        
        for i, elem in enumerate(elements):
            for g in gens:
                # Compute elem * g
                a1, b1, c1 = elem
                a2, b2, c2 = g
                new_a = (a1 + a2) % q
                new_b = (b1 + b2) % q
                new_c = (c1 + c2 + a1 * b2) % q
                neighbor = (new_a, new_b, new_c)
                j = elem_index[neighbor]
                
                if i != j:  # No loops
                    adj[i].add(j)
                    adj[j].add(i)
                    edge = (min(i, j), max(i, j))
                    edges.add(edge)
        
        # Check if graph is k-regular
        for i in range(n):
            if len(adj[i]) != k:
                # Not k-regular, try different generators
                return None
        
        # Check girth using BFS from each vertex
        # We need to verify girth >= g
        def check_girth(adj, n, required_girth):
            # BFS from each vertex to find shortest cycle
            # For each vertex u, do BFS and track parent
            # If we find a vertex w that is already visited, and w is not the parent,
            # then we have a cycle of length dist[u] + dist[w] + 1
            # But this is expensive. Let's use a smarter approach.
            
            # Alternative: Use the fact that in a Cayley graph, the girth is the
            # length of the shortest non-trivial product of generators that equals identity.
            
            # For now, let's do a simple check: BFS from each vertex up to depth required_girth//2
            # and see if we can find a cycle.
            
            for start in range(n):
                # BFS from start
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    if dist[u] >= required_girth // 2:
                        continue
                    
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < required_girth:
                                return False
                        # If v is parent[u] or u is parent[v], it's just the tree edge, not a cycle
            
            return True
        
        if not check_girth(adj, n, g):
            return None
        
        return n, list(edges)
    
    # Try ABS construction first
    try:
        result = abs_construct(k, g)
        if result and result[0] < 1116:  # Better than known best
            return result
    except Exception:
        pass
    
    # Fallback to LUW
    return luw_construct(k, g)
