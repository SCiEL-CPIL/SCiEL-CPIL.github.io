import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k != 4 or g != 10:
        raise ValueError("Only k=4, g=10 is supported")

    # Strategy 1: Try to find the 320-vertex cage via voltage lift of K_4,4 with m=10
    result_320 = try_find_320_cage(deadline)
    if result_320:
        return result_320

    # Strategy 2: Fallback to LUW graph with d=4 (girth 10, 512 vertices)
    return luw_graph(4, 4)

def luw_graph(k, d):
    # LUW graph CD(d, q) with q=k
    # Girth is d+6 if d is even, d+5 if d is odd
    # For g=10, we need d=4 (even) -> girth 10
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

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
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def try_find_320_cage(deadline):
    m = 10
    # Base graph K_4,4
    # Vertices 0-3 (L), 4-7 (R)
    # Edges (l, 4+r)
    
    # Precompute cycles of length 4, 6, 8
    # Represent cycles as list of edges (l, r)
    # Note: The voltage sum depends on the direction of traversal.
    # We define the voltage of edge (l, 4+r) as v[l][r].
    # Traversal l -> 4+r has voltage v[l][r].
    # Traversal 4+r -> l has voltage -v[l][r].
    
    cycles = []
    
    # 4-cycles: l1, r1, l2, r2
    # Walk: l1 -> 4+r1 -> l2 -> 4+r2 -> l1
    # Edges: (l1, r1) fwd, (l2, r1) bwd, (l2, r2) fwd, (l1, r2) bwd
    # Sum: v[l1][r1] - v[l2][r1] + v[l2][r2] - v[l1][r2]
    for l1 in range(4):
        for l2 in range(l1+1, 4):
            for r1 in range(4):
                for r2 in range(r1+1, 4):
                    # Check if this is a valid cycle in K_4,4 (it is)
                    # Constraint: sum != 0, 5 mod 10
                    cycles.append(([(l1, r1, 1), (l2, r1, -1), (l2, r2, 1), (l1, r2, -1)], 4))
                    
    # 6-cycles: l1, r1, l2, r2, l3, r3
    # Walk: l1 -> 4+r1 -> l2 -> 4+r2 -> l3 -> 4+r3 -> l1
    # Edges: (l1, r1) fwd, (l2, r1) bwd, (l2, r2) fwd, (l3, r2) bwd, (l3, r3) fwd, (l1, r3) bwd
    # Sum: v[l1][r1] - v[l2][r1] + v[l2][r2] - v[l3][r2] + v[l3][r3] - v[l1][r3]
    # Constraint: sum != 0 mod 10
    for l1 in range(4):
        for l2 in range(4):
            if l2 == l1: continue
            for l3 in range(4):
                if l3 == l1 or l3 == l2: continue
                for r1 in range(4):
                    for r2 in range(4):
                        if r2 == r1: continue
                        for r3 in range(4):
                            if r3 == r1 or r3 == r2: continue
                            # This generates all 6-cycles, but with duplicates (rotations/reflections)
                            # We just need to check the constraint. Duplicates are fine.
                            cycles.append(([(l1, r1, 1), (l2, r1, -1), (l2, r2, 1), (l3, r2, -1), (l3, r3, 1), (l1, r3, -1)], 6))

    # 8-cycles: l1, r1, l2, r2, l3, r3, l4, r4
    # Walk: l1 -> 4+r1 -> l2 -> 4+r2 -> l3 -> 4+r3 -> l4 -> 4+r4 -> l1
    # Edges: (l1, r1) fwd, (l2, r1) bwd, (l2, r2) fwd, (l3, r2) bwd, (l3, r3) fwd, (l4, r3) bwd, (l4, r4) fwd, (l1, r4) bwd
    # Sum: v[l1][r1] - v[l2][r1] + v[l2][r2] - v[l3][r2] + v[l3][r3] - v[l4][r3] + v[l4][r4] - v[l1][r4]
    # Constraint: sum != 0 mod 10
    for l1 in range(4):
        for l2 in range(4):
            if l2 == l1: continue
            for l3 in range(4):
                if l3 in (l1, l2): continue
                for l4 in range(4):
                    if l4 in (l1, l2, l3): continue
                    for r1 in range(4):
                        for r2 in range(4):
                            if r2 == r1: continue
                            for r3 in range(4):
                                if r3 in (r1, r2): continue
                                for r4 in range(4):
                                    if r4 in (r1, r2, r3): continue
                                    cycles.append(([(l1, r1, 1), (l2, r1, -1), (l2, r2, 1), (l3, r2, -1), (l3, r3, 1), (l4, r3, -1), (l4, r4, 1), (l1, r4, -1)], 8))

    # Backtracking search
    edges_to_assign = [(l, r) for l in range(4) for r in range(4)]
    v = [[0]*4 for _ in range(4)]
    
    def check_constraints():
        for cyc, length in cycles:
            s = 0
            for l, r, sign in cyc:
                s += sign * v[l][r]
            s %= m
            if length == 4:
                if s == 0 or s == 5:
                    return False
            else:
                if s == 0:
                    return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == 16:
            return check_constraints()
        l, r = edges_to_assign[idx]
        for val in range(m):
            v[l][r] = val
            # Pruning: check constraints involving this edge
            # For speed, we can just check all constraints if m is small
            # But 16 vars, 10 values, ~100 constraints. It's fast.
            if check_constraints():
                if backtrack(idx + 1):
                    return True
        return False

    if backtrack(0):
        # Construct lifted graph
        n_lifted = 8 * m
        edges_lifted = set()
        for l in range(4):
            for r in range(4):
                u_base = l
                v_base = 4 + r
                w = v[l][r]
                for i in range(m):
                    u_lifted = u_base * m + i
                    v_lifted = v_base * m + (i + w) % m
                    if u_lifted > v_lifted:
                        u_lifted, v_lifted = v_lifted, u_lifted
                    edges_lifted.add((u_lifted, v_lifted))
        return n_lifted, list(edges_lifted)
    
    return None
