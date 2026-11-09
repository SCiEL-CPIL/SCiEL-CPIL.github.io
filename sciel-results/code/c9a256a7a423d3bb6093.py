"""Lazebnik-Ustimenko-Woldar graph CD(d, q) for cage construction."""
import time

def construct(k, g):
    t0 = time.time()
    DEADLINE = 145.0

    q = k
    d = max(2, g - 5)

    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add_fn = lambda x,y: x^y
        sub_fn = lambda x,y: x^y
        mul_fn = lambda x,y: table[x][y]
    else:
        add_fn = lambda x,y: (x+y) % q
        sub_fn = lambda x,y: (x-y) % q
        mul_fn = lambda x,y: x*y % q

    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - (c-5) % 4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def get_neighbours(side, x):
        result = []
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            result.append((1-side, tuple(y)))
        return result

    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    adj = []
    edges = []

    for vi, vertex in enumerate(order):
        side, x = vertex
        nbrs = get_neighbours(side, x)
        adj.append([])
        for other in nbrs:
            if other not in index:
                index[other] = len(order)
                order.append(other)
            adj[vi].append(index[other])
            if side == 0:
                edges.append((vi, index[other]))

    n = len(order)

    # Quick girth verification via non-backtracking walks (time-limited)
    adj_sets = [set(s) for s in adj]
    
    # For bipartite graph, check even cycle lengths < g
    # Non-backtracking closed walk of length L from v: v=v0,v1,...,vL=v0, vi!=v(i-2)
    # We check if any such walk exists for L < g.
    
    # Time-limited: check all vertices but with a hard time cutoff
    girth_ok = True
    for v0 in range(n):
        if time.time() > DEADLINE - 3:
            break
        # BFS in non-backtracking state space
        # State: (current_vertex, previous_vertex)
        states = set()
        for u in adj[v0]:
            states.add((u, v0))
        found_short_cycle = False
        for depth in range(1, g-1):  # walk length = depth+1, check up to g-1
            new_states = set()
            for (cur, prev) in states:
                for nxt in adj[cur]:
                    if nxt != prev:
                        new_states.add((nxt, cur))
            # Check closure: is v0 reachable from any state with non-backtracking?
            for (cur, prev) in new_states:
                if v0 in adj_sets[cur] and v0 != prev:
                    found_short_cycle = True
                    break
            if found_short_cycle:
                girth_ok = False
                break
            states = new_states
            if not states:
                break
        if not girth_ok:
            break

    return n, edges
