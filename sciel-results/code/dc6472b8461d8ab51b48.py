"""Rewritten program with stronger algorithm for cage graph construction."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def luw_construct(q, d):
        """Lazebnik-Ustimenko-Woldar graph CD(d,q): q-regular bipartite graph with girth >= d+5"""
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y) % q
            sub = lambda x,y: (x-y) % q
            mul = lambda x,y: x*y % q
        
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r = (c-5) % 4
            c0 = c - r
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1-side, tuple(y))
        
        start_v = (0, (None,) + (0,)*d)
        index = {start_v: 0}
        order = [start_v]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    # For LUW, we need d such that d + 5 >= g, so d >= g - 5
    min_d = max(3, g - 5)
    
    # Use the smallest d that gives girth >= g
    n, edges = luw_construct(k, min_d)
    
    return n, edges
# EVOLVE-BLOCK-END
