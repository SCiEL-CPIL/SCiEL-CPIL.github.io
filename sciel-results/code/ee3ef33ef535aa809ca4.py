"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Fallback to seed logic if something goes wrong or time is up
    def _fallback_construct(k, g):
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

    # Target: k=5, g=9
    # Try SL(2, 11) Cayley graph
    if k == 5 and g == 9:
        try:
            # SL(2, 11) construction
            p = 11
            # Matrices as tuples (a, b, c, d) representing [[a, b], [c, d]]
            def mat_mul(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p, (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
            
            def mat_inv(m):
                a, b, c, d = m
                det = (a*d - b*c) % p
                # det should be 1
                inv_det = pow(det, p-2, p)
                return ((d*inv_det) % p, (-b*inv_det) % p, (-c*inv_det) % p, (a*inv_det) % p)
            
            I = (1, 0, 0, 1)
            A = (0, p-1, 1, 0) # [[0, -1], [1, 0]]
            B = (1, 1, 0, 1)   # [[1, 1], [0, 1]]
            NegI = (p-1, 0, 0, p-1) # [[-1, 0], [0, -1]]
            
            # Check A, B, NegI are in SL(2, p)
            # det(A) = 1, det(B) = 1, det(NegI) = 1
            
            generators = [A, mat_inv(A), B, mat_inv(B), NegI]
            
            # Ensure generators are unique and not identity
            gen_set = set()
            for g in generators:
                if g != I:
                    gen_set.add(g)
            # If any generator is I, we have a problem.
            # Also ensure symmetry: if g in S, g^-1 in S.
            # Our construction ensures this.
            
            # Build Cayley graph
            # Elements of SL(2, 11)
            # Generate all elements
            elements = [I]
            elem_set = {I}
            # BFS to generate group
            # Using generators A, B
            queue = [I]
            while queue:
                current = queue.pop(0)
                for gen in [A, B]:
                    new_elem = mat_mul(current, gen)
                    if new_elem not in elem_set:
                        elem_set.add(new_elem)
                        elements.append(new_elem)
                        queue.append(new_elem)
                    new_elem_inv = mat_mul(current, mat_inv(gen))
                    if new_elem_inv not in elem_set:
                        elem_set.add(new_elem_inv)
                        elements.append(new_elem_inv)
                        queue.append(new_elem_inv)
            
            n = len(elements)
            # n should be 1320
            if n != 1320:
                return _fallback_construct(k, g)
                
            # Map elements to indices
            elem_to_idx = {e: i for i, e in enumerate(elements)}
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            edges = []
            for i, e in enumerate(elements):
                for gen in gen_set:
                    neighbor = mat_mul(e, gen)
                    j = elem_to_idx[neighbor]
                    adj[i].append(j)
                    if i < j:
                        edges.append((i, j))
            
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    return _fallback_construct(k, g)
            
            # Check girth
            # BFS from every vertex to find shortest cycle
            # Girth is min over all vertices of shortest cycle through that vertex?
            # Actually, girth is the length of the shortest cycle in the graph.
            # We can BFS from each vertex to find the shortest path back to itself (non-backtracking).
            # Or simpler: BFS from each vertex, record distance.
            # If we find a neighbor at distance d-1, we have a cycle of length d.
            # Since graph is regular, we can just BFS from one vertex? No, cycles can be anywhere.
            # But for a cage, we need global girth.
            # 1320 vertices is small enough to BFS from all.
            
            min_girth = 1000
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = [start]
                found_cycle = False
                while queue and not found_cycle:
                    curr = queue.pop(0)
                    d = dist[curr]
                    # If d is already >= min_girth, no need to go deeper
                    if d >= min_girth:
                        break
                    for nb in adj[curr]:
                        if dist[nb] == -1:
                            dist[nb] = d + 1
                            queue.append(nb)
                        elif nb != start: # Avoid backtracking to parent immediately?
                            # If we found a neighbor that is already visited and not the start (or parent),
                            # we found a cycle.
                            # In a BFS tree, if (curr, nb) is a cross edge, cycle length is d + dist[nb] + 1.
                            # We need to ensure we don't count the edge (curr, parent) as a cycle.
                            # Standard BFS girth:
                            # For each edge (u, v), if dist[u] == dist[v] or |dist[u]-dist[v]|==1...
                            # Simpler: BFS from start. If we encounter a visited node v (not parent), 
                            # cycle length is dist[curr] + dist[v] + 1.
                            # We want the minimum such cycle.
                            pass
            
            # More robust girth check:
            # For each vertex, BFS.
            # Keep track of parent to avoid backtracking.
            min_girth = 1000
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                while queue:
                    curr = queue.pop(0)
                    if dist[curr] >= min_girth:
                        continue
                    for nb in adj[curr]:
                        if dist[nb] == -1:
                            dist[nb] = dist[curr] + 1
                            parent[nb] = curr
                            queue.append(nb)
                        elif nb != parent[curr]:
                            # Cycle found
                            cycle_len = dist[curr] + dist[nb] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                            if min_girth < g:
                                break
                    if min_girth < g:
                        break
                if min_girth < g:
                    break
            
            if min_girth >= g:
                return n, edges
            else:
                return _fallback_construct(k, g)
                
        except Exception:
            return _fallback_construct(k, g)
            
    return _fallback_construct(k, g)
# EVOLVE-BLOCK-END
