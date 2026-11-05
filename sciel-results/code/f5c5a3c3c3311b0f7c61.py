import time
import random
from collections import defaultdict

_DEADLINE = None
def _ok():
    return time.time() < _DEADLINE

def _luw(k, g):
    d = max(2, g - 4)
    q = k
    if q == 4:
        at = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mt = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x,y): return at[x][y]
        def sub(x,y): return at[x][y]
        def mul(x,y): return mt[x][y]
    else:
        def add(x,y): return (x+y) % q
        def sub(x,y): return (x-y) % q
        def mul(x,y): return (x*y) % q
    rule = [None]*(d+1)
    if d>=2: rule[2]=(1,1)
    if d>=3: rule[3]=(2,1)
    if d>=4: rule[4]=(1,2)
    for c in range(5,d+1):
        r=(c-5)%4; c0=c-r
        if r==0: rule[c]=(1,c0-2)
        elif r==1: rule[c]=(c0-1,1)
        elif r==2: rule[c]=(c0,1)
        else: rule[c]=(1,c0+1)
    start=(0,(0,)*(d+1))
    idx={start:0}; order=[start]; queue=[start]
    edges=[]
    for vertex in queue:
        if not _ok(): break
        side,x=vertex; vi=idx[vertex]
        for first in range(q):
            y=[0]*(d+1); y[1]=first
            if side==0:
                for c in range(2,d+1):
                    a,b=rule[c]; y[c]=add(x[c],mul(y[a],x[b]))
            else:
                for c in range(2,d+1):
                    a,b=rule[c]; y[c]=sub(x[c],mul(x[a],y[b]))
            nb=(1-side,tuple(y))
            if nb not in idx:
                idx[nb]=len(order); order.append(nb); queue.append(nb)
            if side==0:
                edges.append((vi,idx[nb]))
    return len(order), edges

def _girth_ok(adj, g, n):
    maxdepth = (g-1)//2
    for s in range(n):
        seen = {}
        cur = [(s, -1)]
        depth = 0
        while cur and depth < maxdepth:
            nxt = []
            for (u,p) in cur:
                for v in adj[u]:
                    if v == p:
                        continue
                    st = (v,u)
                    if st in seen:
                        L = (depth+1) - seen[st]
                        if L < g:
                            return False
                    else:
                        seen[st] = depth+1
                        nxt.append(st)
            cur = nxt
            depth += 1
    return True

def construct(k, g):
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 115.0
    rng = random.Random(12345)
    depth = (g-1)//2
    # Build tree: root has k children, others have (k-1) children.
    parent = [None]
    node_at_depth = {0:[0]}
    for d in range(1, depth+1):
        node_at_depth[d] = []
        for u in node_at_depth[d-1]:
            nchild = k if d == 1 else (k-1)
            for _ in range(nchild):
                nid = len(parent)
                parent.append(u)
                node_at_depth[d].append(nid)
    n_tree = len(parent)
    leaves = node_at_depth[depth]
    L = len(leaves)
    need = k-1
    def root_branch(u):
        while parent[u] != 0:
            u = parent[u]
        return u
    branch = [root_branch(u) for u in leaves]

    def try_build():
        ldeg = [0]*L
        ladj = [set() for _ in range(L)]
        deficit = [need]*L
        perm = list(range(L)); rng.shuffle(perm)
        for i in perm:
            if deficit[i] <= 0: continue
            cand = [j for j in range(L) if j!=i and branch[j]!=branch[i] and deficit[j]>0 and j not in ladj[i]]
            rng.shuffle(cand)
            for j in cand:
                if deficit[i] <= 0: break
                ladj[i].add(j); ladj[j].add(i)
                deficit[i]-=1; deficit[j]-=1
            if deficit[i] > 0:
                return None
        if any(d>0 for d in deficit):
            return None
        return ladj

    ladj = None
    for _ in range(5000):
        if not _ok(): break
        ladj = try_build()
        if ladj is not None:
            break
    if ladj is None:
        return _luw(k,g)

    # Build and verify
    def build(ladj):
        adj = [set() for _ in range(n_tree)]
        for u in range(1, n_tree):
            p = parent[u]; adj[u].add(p); adj[p].add(u)
        for i in range(L):
            u = leaves[i]
            for j in ladj[i]:
                v = leaves[j]; adj[u].add(v); adj[v].add(u)
        return adj

    adj = build(ladj)
    if not (all(len(adj[u])==k for u in range(n_tree)) and _girth_ok([list(a) for a in adj], g, n_tree)):
        ok = False
        for _ in range(2000):
            if not _ok(): break
            ladj = try_build()
            if ladj is None: continue
            adj = build(ladj)
            if all(len(adj[u])==k for u in range(n_tree)) and _girth_ok([list(a) for a in adj], g, n_tree):
                ok = True; break
        if not ok:
            return _luw(k,g)

    edges = []
    for u in range(n_tree):
        for v in adj[u]:
            if u < v:
                edges.append((u,v))
    return n_tree, edges
