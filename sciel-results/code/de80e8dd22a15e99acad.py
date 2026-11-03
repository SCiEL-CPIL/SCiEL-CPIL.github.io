import time
import sys
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k != 3:
        return 0, []

    q = k

    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    def build_luw(q, d):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x,y): return x^y
            def sub(x,y): return x^y
            def mul(x,y): return table[x][y]
        else:
            def add(x,y): return (x+y)%q
            def sub(x,y): return (x-y)%q
            def mul(x,y): return x*y%q

        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r = (c-5)%4
            c0 = c-r
            if r==0: rule[c]=(1,c0-2)
            elif r==1: rule[c]=(c0-1,1)
            elif r==2: rule[c]=(c0,1)
            else: rule[c]=(1,c0+1)

        index = {}
        order = []
        edges = []
        start = (0, (None,)+(0,)*d)
        index[start] = 0
        order.append(start)

        for i, vertex in enumerate(order):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side==0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                neighbor = (1-side, tuple(y))
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                if side==0:
                    edges.append((index[vertex], index[neighbor]))

        return len(order), edges

    def check_girth(n, edges, k, g):
        adj = collections.defaultdict(list)
        for u,v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        for sv in range(n):
            visited = set()
            queue = collections.deque()
            for nb in adj[sv]:
                s = (nb, sv, 1)
                visited.add(s)
                queue.append(s)
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= g-1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    ns = (nb, node, depth+1)
                    if ns in visited:
                        return False
                    visited.add(ns)
                    queue.append(ns)
        return True

    d = 3
    while girth_of_d(d) < g:
        d += 1

    for attempt in range(d, d+5):
        if time.time() > deadline - 30:
            break
        n, edges = build_luw(q, attempt)
        unique_edges = list(set(edges))
        if check_girth(n, unique_edges, k, g):
            return n, unique_edges

    return 0, []
