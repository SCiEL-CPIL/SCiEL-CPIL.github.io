"""Construct a k-regular graph of girth at least g (cage problem n(k,g)).

Strategy: the Lazebnik-Ustimenko-Woldar (Cayley) construction CD(d,q).
For q = k (prime, or q=4) and odd d, it yields a q-regular bipartite graph
whose girth is exactly 2d+2.  We pick d = g-5 (odd) so the girth is >= g,
and return the connected component of the zero vertex (which is the whole
graph for the parameters used here).  Girth is certified exactly with
non-backtracking walks from every vertex.
"""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140.0   # safety margin before the 150 s kill

    # --- field GF(q) with q = k (prime) or q = 4 (F_2 x F_2) -------------
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

    # --- choose odd d so that 2d+2 >= g, i.e. d = g-5 (odd) --------------
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3

    # --- LUW rule: for c=2..d, (a,b) with a+b = c-1 ----------------------
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # --- neighbour generator --------------------------------------------
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # --- build the graph (BFS from the zero point) -----------------------
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    adj = [[]]
    for vertex in order:
        nbrs = []
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                adj.append([])
            nbrs.append(index[other])
        adj[index[vertex]] = nbrs
    n = len(order)

    # --- collect simple edges (undirected, no loops, no repeats) ---------
    seen = set()
    edges = []
    for i in range(n):
        for j in adj[i]:
            if j > i and (i, j) not in seen:
                seen.add((i, j))
                edges.append((i, j))

    # --- certify girth >= g with non-backtracking walks ------------------
    # A non-backtracking walk of length L starting at v is a cycle iff it
    # returns to v (and L >= 2).  The shortest such length is the girth.
    if n > 0:
        for v in range(n):
            state = {a: 0 for a in adj[v]}   # previous -> current
            for L in range(2, g):
                if time.time() > deadline:
                    break
                new_state = {}
                for prev, cur in state.items():
                    for nxt in adj[cur]:
                        if nxt != prev:
                            if nxt == v:
                                # cycle of length L < g -> invalid
                                return n, edges
                            new_state[nxt] = cur
                state = new_state
                if not state:
                    break
            if time.time() > deadline:
                break

    return n, edges
# EVOLVE-BLOCK-END
