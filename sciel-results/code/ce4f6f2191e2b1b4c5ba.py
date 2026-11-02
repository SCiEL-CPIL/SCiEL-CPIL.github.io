import time
import random
import heapq
import sys
import numpy as np

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

def _construct_psl2_11_cayley():
    """
    Constructs a 5-regular Cayley graph on PSL(2, 11) (n=660).
    Tries to find a generating set of 5 involutions that yields girth >= 9.
    """
    # PSL(2, 11) construction
    # Elements are 2x2 matrices over GF(11) with determinant 1, modulo scalars.
    # Since 11 is prime, GF(11) is just Z_11.
    # We represent elements as tuples (a, b, c, d) for matrix [[a, b], [c, d]].
    # Identity: (1, 0, 0, 1)
    
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % 11, (a1*b2 + b1*d2) % 11, (c1*a2 + d1*c2) % 11, (c1*b2 + d1*d2) % 11)
        
    def mat_inv(m):
        a, b, c, d = m
        # det = ad - bc = 1
        # inv = [[d, -b], [-c, a]]
        return (d, (-b) % 11, (-c) % 11, a)
        
    # Generate all elements
    elements = []
    seen = set()
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    if (a*d - b*c) % 11 == 1:
                        # Check if scalar multiple of identity?
                        # PSL(2,11) is SL(2,11) / {I, -I}.
                        # So we identify M with -M.
                        # -M = (-a, -b, -c, -d) % 11
                        # We can just take the "canonical" representative.
                        # Or just generate SL(2,11) and quotient.
                        # SL(2,11) has order 1320. PSL has 660.
                        elements.append((a, b, c, d))
                        
    # This generates 1320 elements.
    # We need to quotient by -I.
    # Let's map each element to a canonical form.
    # If (a,b,c,d) and (-a,-b,-c,-d) are both in the list, we pick one.
    # We'll just use the list of 1320 and treat M and -M as same?
    # No, easier: generate the 660 elements directly.
    # Condition: a*d - b*c = 1.
    # And we identify (a,b,c,d) with (-a,-b,-c,-d).
    # Let's just generate all 1320 and then pick one from each pair.
    
    sl_elements = []
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    if (a*d - b*c) % 11 == 1:
                        sl_elements.append((a, b, c, d))
                        
    # Quotient
    psl_elements = []
    used = set()
    for m in sl_elements:
        if m in used:
            continue
        psl_elements.append(m)
        # Mark m and -m as used
        used.add(m)
        neg_m = ((-m[0])%11, (-m[1])%11, (-m[2])%11, (-m[3])%11)
        used.add(neg_m)
        
    n = len(psl_elements)
    if n != 660:
        return None
        
    # Map element to index
    idx = {m: i for i, m in enumerate(psl_elements)}
    
    # Find involutions
    involutions = []
    for m in psl_elements:
        if mat_mul(m, m) == (1, 0, 0, 1) or mat_mul(m, m) == (10, 0, 0, 10): # -I
            # In PSL, M^2 = I or M^2 = -I (which is I in PSL)
            # So M^2 is I or -I in SL.
            # If M^2 = -I, then in PSL, M^2 = I.
            involutions.append(m)
            
    # We need 5 involutions that generate the group and form a symmetric set.
    # Randomly pick 5 involutions.
    # Check girth.
    
    def get_girth(generators):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            for g in generators:
                # e_i * g
                prod = mat_mul(psl_elements[i], g)
                # Find index of prod
                # Note: prod might be -psl_elements[j]?
                # No, we stored canonical reps.
                # We need to map prod to canonical rep.
                # If prod is in idx, use it.
                # Else, use -prod.
                if prod in idx:
                    j = idx[prod]
                else:
                    neg_prod = ((-prod[0])%11, (-prod[1])%11, (-prod[2])%11, (-prod[3])%11)
                    j = idx[neg_prod]
                adj[i].append(j)
                
        # Check for multi-edges (should be none if generators are distinct and not 1)
        # Check for loops (generators not 1)
        
        # BFS for girth
        # Since n=660, BFS from each vertex is feasible.
        min_girth = 100
        for start in range(n):
            visited = [-1] * n
            queue = [(start, 0)]
            visited[start] = 0
            while queue:
                node, dist = queue.pop(0)
                if dist > 10: # We only care about girth < 9
                    continue
                for neighbor in adj[node]:
                    if visited[neighbor] == -1:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
                    else:
                        # Found a cycle
                        # Cycle length = dist + 1 + visited[neighbor]
                        # This is not exact for non-BFS trees, but gives a bound.
                        # Better: standard BFS for shortest cycle.
                        pass
                        
        # Simpler: check if there is a cycle of length < 9.
        # We can just run BFS up to depth 4 from each vertex.
        # If we reach a neighbor of the start, we have a cycle.
        
        for start in range(n):
            visited = {start: 0}
            queue = [(start, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist >= 4: # If dist=4, neighbor is dist 5, cycle 5? No.
                    # We want to detect cycles of length 3, 4, 5, 6, 7, 8.
                    # Max depth 4.
                    continue
                for neighbor in adj[node]:
                    if neighbor == start and dist + 1 >= 3:
                        return dist + 1
                    if neighbor in visited:
                        if visited[neighbor] + dist + 1 < 9:
                            return visited[neighbor] + dist + 1
                    else:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
                        
        return 100 # No short cycle found
        
    # Try random sets of 5 involutions
    random.seed(42)
    for _ in range(100):
        if len(involutions) < 5:
            break
        gens = random.sample(involutions, 5)
        # Check if they generate the group?
        # For now, just check girth.
        girth = get_girth(gens)
        if girth >= 9:
            # Construct edges
            edges = []
            for i in range(n):
                for g in gens:
                    prod = mat_mul(psl_elements[i], g)
                    if prod in idx:
                        j = idx[prod]
                    else:
                        neg_prod = ((-prod[0])%11, (-prod[1])%11, (-prod[2])%11, (-prod[3])%11)
                        j = idx[neg_prod]
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    return None

def construct(k, g):
    if k == 5 and g == 9:
        # Try PSL(2,11) Cayley graph
        res = _construct_psl2_11_cayley()
        if res is not None:
            return res
            
        # Fallback to LUW
        return _luw_construct(k, g)
        
    # Fallback to LUW for other parameters
    return _luw_construct(k, g)
