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

    def check_girth(n, edges, required_girth):
        """Check if graph has no cycle shorter than required_girth using non-backtracking BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do a BFS tracking parents to avoid backtracking
        # A non-backtracking walk of length L from v corresponds to a path in the line graph
        # But a simpler approach: BFS from each vertex, but when expanding, don't go back to parent.
        # If we find a vertex at distance d that was already visited at distance d' < d, 
        # and the paths don't share the immediate predecessor, we might have a cycle.
        # Actually, the standard way to check girth is:
        # For each start vertex s, do a BFS. If we encounter a back-edge (u,v) where u and v are both 
        # visited and neither is the parent of the other, the cycle length is dist[u] + dist[v] + 1.
        # However, this can miss some cycles or count them multiple times. 
        # A more robust method for small girth: 
        # Non-backtracking walks. A cycle of length L exists iff there is a non-backtracking walk of length L 
        # that returns to the start. 
        # Given the time constraint, let's use the standard BFS girth check which is O(n * (n+m)).
        # For n ~ 512, this is fine.
        
        min_girth = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = [s]
            dist[s] = 0
            parent[s] = -2
            while queue:
                curr = queue.pop(0)
                if dist[curr] >= required_girth:
                    continue
                for nbr in adj[curr]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[curr] + 1
                        parent[nbr] = curr
                        queue.append(nbr)
                    elif nbr != parent[curr]:
                        cycle_len = dist[curr] + dist[nbr] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < required_girth:
                                return min_girth
        return min_girth

    if k == 4 and g == 10:
        # Try smaller LUW first (d=4 -> n=512)
        # If girth is sufficient, return it.
        n_test, edges_test = get_luw_graph(4)
        
        # Quick girth check
        girth = check_girth(n_test, edges_test, g)
        
        if girth >= g:
            return n_test, edges_test

        # Fallback to standard LUW (d=5 -> n=2048)
        return get_luw_graph(5)
    
    # Fallback to LUW
    return get_luw_graph()
