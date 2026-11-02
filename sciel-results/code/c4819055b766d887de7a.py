"""Construct a 5-regular graph of girth at least 9 with as few vertices as possible."""
import time
import random
from collections import defaultdict

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110
    
    def build_random_regular(k, n, seed):
        """Configuration model for k-regular graph."""
        rng = random.Random(seed)
        stubs = list(range(n * k))
        # Assign stubs to vertices
        stub_owner = [i % n for i in range(n * k)]
        stubs_idx = list(range(n * k))
        rng.shuffle(stubs_idx)
        adj = [set() for _ in range(n)]
        i = 0
        attempts = 0
        while i < len(stubs_idx) - 1:
            u_idx = stubs_idx[i]
            u = u_idx % n
            j = rng.randint(i + 1, len(stubs_idx) - 1)
            v_idx = stubs_idx[j]
            v = v_idx % n
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                # Swap to remove used stubs
                stubs_idx[i] = stubs_idx[-1]
                stubs_idx[j] = stubs_idx[-2]
                stubs_idx.pop()
                stubs_idx.pop()
                attempts = 0
            else:
                attempts += 1
                if attempts > n * k * 10:
                    return None
        # Verify
        for v in range(n):
            if len(adj[v]) != k:
                return None
        return [sorted(adj[v]) for v in range(n)]
    
    def has_short_cycle(adj, n, target):
        """Check for any cycle shorter than target. Returns (found, cycle_len)."""
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            for idx in range(len(queue)):
                u = queue[idx]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != queue[idx-1] if idx > 0 else True:
                        # Check if this creates a cycle
                        pass
            # More careful BFS for girth
        # Use BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for idx in range(len(queue)):
                u = queue[idx]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target:
                            return True, cycle_len
        return False, target
    
    def luw_fallback(k, g):
        """LUW construction - always valid but large."""
        q = k
        # For bipartite girth >= g+1 (if g odd) or >= g (if g even)
        # LUW d gives girth >= d+5
        if g % 2 == 1:
            need_girth = g + 1
        else:
            need_girth = g
        d = max(3, need_girth - 5)
        
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
            c0 = c - (c-5) % 4
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
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
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
    
    # Try random construction for various sizes
    best = None
    for n in range(450, 900, 2):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            adj = build_random_regular(k, n, seed)
            if adj is None:
                continue
            has_cycle, clen = has_short_cycle(adj, n, g)
            if not has_cycle:
                best = (n, adj)
                found = True
                break
        if found:
            break
    
    if best is not None:
        n, adj = best
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges
    
    # Fallback to LUW
    return luw_fallback(k, g)
