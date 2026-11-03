import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 20s margin for safety

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def construct_lw(k, g):
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

    # Specific case k=5, g=9
    if k == 5 and g == 9:
        # Try SL(2, 11) construction
        def construct_sl2_11():
            # Generate SL(2, 11)
            # Matrices [[a, b], [c, d]] with ad - bc = 1 mod 11
            elements = []
            elem_index = {}
            for a in range(11):
                for b in range(11):
                    for c in range(11):
                        # d = (1 + bc) * a^{-1} if a != 0
                        if a == 0:
                            # -bc = 1 => bc = -1 = 10
                            # If b=0, no solution. If b!=0, c = 10 * b^{-1}
                            if b == 0:
                                continue
                            c_val = (10 * pow(b, 10, 11)) % 11
                            if c == c_val:
                                d = 0
                                m = (a, b, c, d)
                                if m not in elem_index:
                                    elem_index[m] = len(elements)
                                    elements.append(m)
                        else:
                            a_inv = pow(a, 10, 11)
                            d = ((1 + b * c) % 11) * a_inv % 11
                            m = (a, b, c, d)
                            if m not in elem_index:
                                elem_index[m] = len(elements)
                                elements.append(m)
            
            # Verify count
            if len(elements) != 1320:
                return None

            # Define multiplication
            def mul_mat(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                return ((a1*a2 + b1*c2) % 11, (a1*b2 + b1*d2) % 11, (c1*a2 + d1*c2) % 11, (c1*b2 + d1*d2) % 11)

            # Find involutions (order 2)
            # m^2 = I => trace = 0
            involutions = []
            for m in elements:
                a, b, c, d = m
                if (a + d) % 11 == 0:
                    # Check m^2 = I
                    m2 = mul_mat(m, m)
                    if m2 == (1, 0, 0, 1):
                        involutions.append(m)
            
            # We need to pick 5 involutions
            # Random search
            random.seed(42)
            n_inv = len(involutions)
            
            # Precompute neighbors for fast girth check?
            # We will build the graph for a candidate set and check girth.
            
            def check_girth(generators):
                # Build adjacency list
                adj = {i: [] for i in range(len(elements))}
                for i, m in enumerate(elements):
                    for g in generators:
                        prod = mul_mat(m, g)
                        j = elem_index[prod]
                        adj[i].append(j)
                
                # Check girth >= 9
                # Non-backtracking walks of length up to 8
                for u in range(len(elements)):
                    # BFS on state (node, prev_node)
                    visited = set()
                    queue = []
                    for v in adj[u]:
                        queue.append((v, u, 1))
                        visited.add((v, u))
                    
                    while queue:
                        curr, prev, depth = queue.pop(0)
                        if curr == u:
                            return False
                        if depth >= 8:
                            continue
                        for next_node in adj[curr]:
                            if next_node == prev:
                                continue
                            state = (next_node, curr)
                            if state not in visited:
                                visited.add(state)
                                queue.append((next_node, curr, depth + 1))
            return None # Placeholder, logic below

            # We need to return edges, not just check.
            # Let's try a few random sets.
            for _ in range(100): # Try 100 random sets
                if time.time() > deadline - 10:
                    break
                gens = random.sample(involutions, 5)
                
                # Quick check: do they generate the group?
                # Optional, but if they don't, graph is disconnected.
                # Disconnected is allowed if girth is ok.
                # But let's assume we want connected.
                
                # Build graph
                adj = [set() for _ in range(len(elements))]
                for i, m in enumerate(elements):
                    for g in gens:
                        prod = mul_mat(m, g)
                        j = elem_index[prod]
                        adj[i].add(j)
                
                # Check girth
                girth_ok = True
                for u in range(len(elements)):
                    visited = set()
                    queue = []
                    for v in adj[u]:
                        queue.append((v, u, 1))
                        visited.add((v, u))
                    
                    while queue:
                        curr, prev, depth = queue.pop(0)
                        if curr == u:
                            girth_ok = False
                            break
                        if depth >= 8:
                            continue
                        for next_node in adj[curr]:
                            if next_node == prev:
                                continue
                            state = (next_node, curr)
                            if state not in visited:
                                visited.add(state)
                                queue.append((next_node, curr, depth + 1))
                    if not girth_ok:
                        break
                
                if girth_ok:
                    # Found a valid graph
                    edges = []
                    for i in range(len(elements)):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return len(elements), edges
            
            return None

        result = construct_sl2_11()
        if result:
            return result

    # Fallback to LUW
    return construct_lw(k, g)
