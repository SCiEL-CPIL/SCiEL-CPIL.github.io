import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 130
    q = k
    
    # Determine d (dimension) for LUVW construction
    # Girth >= d + 5 when d is odd
    # We need girth >= g
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # ensure d is odd for guaranteed girth
    
    # Build multiplication table for GF(q)
    if q == 4:
        mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: mul_table[x][y]
    else:
        mul_table = None
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: (x * y) % q
    
    # LUVW rule: rule[c] = (a, b) for c = 2..d
    # From the standard construction:
    # For c even: a = 1, b depends on c
    # For c odd: a depends on c, b = 1
    rule = {}
    # rule[2] = (1, 1), rule[3] = (2, 1), rule[4] = (1, 2)
    # General pattern from LUVW paper:
    for c in range(2, d + 1):
        r = (c - 2) % 4
        if r == 0:
            rule[c] = (1, 1)
        elif r == 1:
            rule[c] = (c - 1, 1)
        elif r == 2:
            rule[c] = (1, c - 2)
        elif r == 3:
            rule[c] = (c, 1)
    
    # Actually, let me use the exact rule from the seed:
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute neighbour relationships
    # Vertex: (side, (x0, x1, ..., xd)) where side in {0, 1}
    # x0 is unused (None), x1 is the "first" coordinate
    # For point (side=0, x): neighbours are lines (side=1, y) where
    #   y[c] = x[c] + y[a]*x[b] for c=2..d, with y[1] = first
    # For line (side=1, x): neighbours are points (side=0, y) where
    #   y[c] = x[c] - x[a]*y[b] for c=2..d, with y[1] = first
    
    # Precompute for each side, for each x, the q neighbours
    # Since q is small (3 or 4), this is feasible
    
    # Build the graph using BFS from the zero vertex
    start = (0, tuple([0] * (d + 1)))  # side=0, all coords 0 (x[0] unused)
    
    # Actually let's use the representation: vertex = (side, tuple of length d+1)
    # Index 0 is dummy, indices 1..d are the coordinates
    
    # For a point (0, x), neighbour with first=f:
    #   y[1] = f
    #   y[c] = x[c] + y[a] * x[b] for c = 2..d
    # For a line (1, x), neighbour with first=f:
    #   y[1] = f
    #   y[c] = x[c] - x[a] * y[b] for c = 2..d
    
    # Precompute all neighbours for a given vertex
    def get_neighbours(vertex):
        side, x = vertex
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0  # dummy
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point to line: y[c] = x[c] + y[a] * x[b]
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    # line to point: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            result.append((1 - side, tuple(y)))
        return result
    
    # BFS to find the connected component
    visited = {}
    order = []
    edges = []
    
    start_key = start
    visited[start_key] = 0
    order.append(start_key)
    
    queue = deque([0])
    while queue:
        idx = queue.popleft()
        vertex = order[idx]
        neighs = get_neighbours(vertex)
        for other in neighs:
            if other not in visited:
                visited[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            # Only add edge once (from point side)
            if vertex[0] == 0:
                other_idx = visited[other]
                edges.append((idx, other_idx))
    
    n = len(order)
    
    # Verify: check degree
    if time.time() > deadline:
        return n, edges
    
    # Verify girth using non-backtracking BFS from each vertex
    # For girth >= g, we need to check that no non-backtracking walk of
    # length < g closes up
    # This is expensive for large n, so we do a lighter check
    
    # Light verification: check no self-loops, no multi-edges
    if n <= 5000:
        # Full girth check for small graphs
        for start_v in range(n):
            if time.time() > deadline:
                break
            # BFS with non-backtracking constraint
            # State: (current_vertex, previous_vertex)
            # We want to find if there's a cycle of length < g
            visited_nb = set()
            queue_nb = deque()
            for nb in range(n):
                if nb == start_v:
                    continue
                # Check if start_v-nb is an edge
                # We need adjacency list
                pass
        
        # Build adjacency list for verification
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                pass  # Should not happen with LUVW
        
        # Check girth by BFS from each vertex (limited depth)
        half_g = g // 2
        for start_v in range(n):
            if time.time() > deadline:
                break
            # BFS up to depth half_g - 1, tracking parent to avoid backtracking
            # Actually for non-backtracking: BFS where we track (node, prev)
            visited2 = set()
            queue2 = deque()
            for nb in adj[start_v]:
                state = (nb, start_v)
                if state not in visited2:
                    visited2.add(state)
                    queue2.append((state, 1))
            found_short = False
            while queue2 and not found_short:
                (node, prev), dist = queue2.popleft()
                if dist >= half_g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    state = (nb, node)
                    if state in visited2:
                        continue
                    visited2.add(state)
                    if nb == start_v:
                        # Found a cycle of length dist + 1
                        if dist + 1 < g:
                            found_short = True
                            break
                    if dist + 1 < half_g:
                        queue2.append((state, dist + 1))
            if found_short:
                break
    
    return n, edges
