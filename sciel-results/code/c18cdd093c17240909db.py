import time
import sys
import collections

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # We assume k=3 and g=16 based on the task description.
    # If k != 3, we cannot use LUW directly for this specific optimization.
    # The task says "evaluator passes this task's degree k and girth g".
    # If k != 3, we might fail or return a generic graph.
    # Given the context, k=3 is the target.
    
    if k != 3:
        # Fallback: Return a simple cycle if k=2? No, must be k-regular.
        # For now, if k != 3, we try to find a prime power q=k if possible, 
        # but LUW requires q to be a prime power.
        # If k is not a prime power, LUW fails.
        # We will return an empty graph which scores 0, or try a different construction.
        # For this specific task, k=3 is expected.
        return 0, []

    q = k # q=3
    
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6
            
    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    if time.time() > deadline:
        return 0, []

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c>=5: cycle through patterns
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Generate vertices
    index = {}
    order = []
    edges = []
    
    # Start vertex: side 0, x = (None, 0, ..., 0)
    # The tuple represents (x0, x1, ..., xd) where x0 is None/placeholder?
    # In the code: y = [None, first] + [0]*(d-1)
    # So y[0] is None, y[1] is first, y[2..d] are 0.
    # This matches the structure.
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    completed = False
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        # Generate neighbors
        for first in range(q):
            # y[0] is None, y[1] is 'first', y[2..d] are computed
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            # Only add edge if it goes forward in the BFS tree or is a cross edge?
            # The LUW graph is bipartite.
            # We add edges from side 0 to side 1.
            # Since we iterate over all vertices, we might add duplicates if we're not careful.
            # However, the graph is 3-regular.
            # If we add an edge when we discover a new neighbor, we miss edges to already discovered neighbors?
            # No, in a BFS, when we process a vertex, we generate all its neighbors.
            # If a neighbor is already in the index, we should still add the edge?
            # Yes, because the graph might have chords or the BFS tree doesn't cover all edges.
            # But LUW is a specific algebraic construction.
            # The edges are defined by the local rules.
            # Every vertex has exactly q neighbors on the other side.
            # So we should add the edge for every `first` in `range(q)`.
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
    
    n = len(order)
    
    # Check if we finished in time
    if time.time() > deadline:
        return 0, []
    
    # The LUW construction should produce a simple graph.
    # We return the edges.
    # To be safe against duplicate edges in the list (which shouldn't happen if the graph is simple and we add each edge once from the side 0 perspective),
    # we can convert to a set.
    # However, converting 500k edges to a set is fast.
    
    unique_edges = list(set(edges))
    
    return n, unique_edges
