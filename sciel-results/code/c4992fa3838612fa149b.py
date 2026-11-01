import time
import random
import itertools
import math

# EVOLVE-BLOCK-START

def _verify_girth(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g."""
    if n <= 0 or not edges:
        return False
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj[u].append(v)
        adj[v].append(u)
    
    for v in range(n):
        if len(adj[v]) != k:
            return False
    
    # Check girth using BFS from each vertex
    for s in range(n):
        depth = [-1] * n
        parent = [-1] * n
        depth[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = depth[u] + depth[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _generate_petersen():
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i + 2) % 5]))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges

def _generate_heawood():
    lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
    edges = []
    for i, line in enumerate(lines):
        for p in line:
            edges.append((p, 7 + i))
    return 14, edges

def _luw_construct(k, g):
    if g % 2 == 0:
        return None
    q = k
    d = (g - 1) // 2
    if d < 1:
        return None
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x,y): return x^y
        def sub(x,y): return x^y
        def mul(x,y): return table[x][y]
    else:
        def add(x,y): return (x+y)%q
        def sub(x,y): return (x-y)%q
        def mul(x,y): return (x*y)%q
    
    rule = {}
    for c in range(1, d+1):
        if c == 1:
            rule[c] = (0,1)
        elif c == 2:
            rule[c] = (1,1)
        else:
            rule[c] = (1,c-1)
    
    vertices = []
    for side in range(2):
        for x in itertools.product(range(q), repeat=d):
            vertices.append((side, x))
    n = len(vertices)
    vid = {v:i for i,v in enumerate(vertices)}
    
    edges = []
    for side, x in vertices:
        for first in range(q):
            y = [0]*(d+1)
            y[1] = first
            if side == 0:
                for c in range(2, d+1):
                    a,b = rule[c]
                    y[c] = add(x[c-1], mul(y[a-1], x[b-1]))
            else:
                for c in range(2, d+1):
                    a,b = rule[c]
                    y[c] = sub(x[c-1], mul(x[a-1], y[b-1]))
            nbr = (1-side, tuple(y[1:]))
            if nbr in vid:
                u = vid[(side,x)]
                v = vid[nbr]
                if u < v:
                    edges.append((u,v))
    
    edges = list(set(edges))
    return n, edges

def _random_k_regular(k, g, n, deadline):
    for _ in range(5):
        if time.time() > deadline:
            return None
        stubs = []
        for v in range(n):
            stubs.extend([v]*k)
        random.shuffle(stubs)
        adj = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u,v = stubs[i], stubs[i+1]
            if u==v or (min(u,v),max(u,v)) in adj:
                valid=False; break
            adj.add((min(u,v),max(u,v)))
        if not valid: continue
        edges = list(adj)
        if _verify_girth(n, edges, k, g):
            return n, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if (k,g)==(3,5):
        return _generate_petersen()
    if (k,g)==(3,6):
        return _generate_heawood()
    
    if g%2==1:
        res = _luw_construct(k,g)
        if res:
            n,edges = res
            if _verify_girth(n,edges,k,g):
                return n,edges
    
    if g%2==0:
        res = _luw_construct(k,g+1)
        if res:
            n,edges = res
            if _verify_girth(n,edges,k,g):
                return n,edges
    
    d = g//2 if g%2==0 else (g-1)//2
    if g%2==0:
        moore = k*sum((k-1)**i for i in range(d))
    else:
        moore = 1+k*sum((k-1)**i for i in range(d))
    
    for n in range(max(moore,2*k,10), max(moore,2*k,10)+30, 2):
        if time.time()>deadline: break
        if n*k%2!=0: continue
        res = _random_k_regular(k,g,n,deadline)
        if res: return res
    
    return 0,[]

# EVOLVE-BLOCK-END
