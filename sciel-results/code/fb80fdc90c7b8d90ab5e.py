import time
import random
import sys

def construct(k, g):
    deadline = time.time() + 140.0
    
    # Fallback: LUW construction
    def get_luw_graph(target_d=None):
        if target_d is None:
            min_d_odd = max(3, g - 5)
            if min_d_odd % 2 == 0:
                min_d_odd += 1
            min_d_even = max(4, g - 4)
            if min_d_even % 2 == 1:
                min_d_even += 1
            target_d = min(min_d_odd, min_d_even)
        d = target_d
        q = k
        
        if q == 4:
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0: a, b = 1, c0 - 2
            elif r == 1: a, b = c0 - 1, 1
            elif r == 2: a, b = c0, 1
            else: a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        def compute_neighbours(vertex):
            side, x = vertex
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                result.append((1 - side, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            if time.time() > deadline:
                break
            for other in compute_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try to find the Foster graph (n=320) using voltage graph search
    if k == 4 and g == 10:
        # Try smaller LUW first (d=4 -> n=512)
        # If girth is sufficient, return it.
        n_test, edges_test = get_luw_graph(4)
        # Quick girth check on a subset or full BFS if fast
        # For n=512, full BFS is fast.
        adj = [[] for _ in range(n_test)]
        for u, v in edges_test:
            adj[u].append(v)
            adj[v].append(u)
        
        girth = g + 1
        for start in range(n_test):
            dist = [-1] * n_test
            parent = [-1] * n_test
            queue = [start]
            dist[start] = 0
            parent[start] = -2
            while queue:
                curr = queue.pop(0)
                if dist[curr] > g:
                    break
                for nbr in adj[curr]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[curr] + 1
                        parent[nbr] = curr
                        queue.append(nbr)
                    elif nbr != parent[curr]:
                        cycle_len = dist[curr] + dist[nbr] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                            if girth < g:
                                break
            if girth < g:
                break
        
        if girth >= g:
            return n_test, edges_test

        # Fallback to standard LUW (d=5 -> n=2048)
        return get_luw_graph(5)
    
    # Fallback to LUW
    return get_luw_graph()
