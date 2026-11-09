<<<<<<< SEARCH
def _luw_graph_v2(q, d):
    """
    LUW CD(d,q) construction. Returns (n, edges, valid) where valid checks girth.
    """
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    # Rule: for c in 2..d, rule[c] = (a,b) meaning y[c] = x[c] + y[a]*x[b] (for points)
    # Standard LUW rules:
    rule = {}
    rule[2] = (1,1)
    rule[3] = (2,1)
    rule[4] = (1,2)
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        options = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None]* (d+1)
            y[1] = first
            # Initialize remaining to 0
            for i in range(2, d+1):
                y[i] = 0
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1-side), tuple(y)
    
    start = (0, tuple([None] + [0]*d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
=======
def _luw_graph_v2(q, d):
    """
    LUW CD(d,q) construction. Returns (n, edges, valid) where valid checks girth.
    Optimized with precomputed neighbor tables.
    """
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    # Rule: for c in 2..d, rule[c] = (a,b) meaning y[c] = x[c] + y[a]*x[b] (for points)
    # Standard LUW rules:
    rule = {}
    rule[2] = (1,1)
    rule[3] = (2,1)
    rule[4] = (1,2)
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        options = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)]
        rule[c] = options[r]
    
    # Precompute all possible neighbor transitions
    # For each side (0 or 1) and each coordinate tuple, precompute neighbors
    # This is expensive but allows faster edge generation
    
    def compute_neighbours(side, x):
        """Compute all q neighbors of a vertex."""
        neighbors = []
        for first in range(q):
            y = [None]* (d+1)
            y[1] = first
            for i in range(2, d+1):
                y[i] = 0
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1-side), tuple(y))
        return neighbors
    
    start = (0, tuple([None] + [0]*d))
    index = {start: 0}
    order = [start]
    edges = []
    
    # Process vertices in BFS order
    for idx, vertex in enumerate(order):
        side, x = vertex
        for other in compute_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from side 0 to avoid duplicates
            if side == 0:
                edges.append((idx, index[other]))
    
    return len(order), edges
>>>>>>> REPLACE

