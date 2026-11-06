import time
import random
from itertools import combinations

_DEADLINE = 105.0
_START = 0.0

_CUBE_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_CUBE_ADJ = [set() for _ in range(8)]
for _a,_b in _CUBE_EDGES:
    _CUBE_ADJ[_a].add(_b); _CUBE_ADJ[_b].add(_a)

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _subset_has_q3(vertices, adj):
    # adj: set of (u,v) with u<v
    n = len(vertices)
    # local edges
    le = [ [False]*n for _ in range(n) ]
    cnt = 0
    for i in range(n):
        for j in range(i+1,n):
            a,b = vertices[i], vertices[j]
            if (a,b) in adj or (b,a) in adj:
                le[i][j]=le[j][i]=True
                cnt+=1
    if cnt < 12:
        return False
    # backtracking assignment cube->local
    used = [False]*n
    assign = [-1]*n
    # order cube vertices by degree desc (all 3) - fixed
    order = list(range(8))
    def bt(pos):
        if pos == 8:
            return True
        c = order[pos]
        for l in range(n):
            if used[l]:
                continue
            ok = True
            for nb in _CUBE_ADJ[c]:
                if assign[nb] != -1:
                    a = assign[nb]
                    if not le[a][l]:
                        ok=False; break
            if not ok:
                continue
            used[l]=True; assign[c]=l
            if bt(pos+1):
                return True
            used[l]=False; assign[c]=-1
        return False
    return bt(0)

def _would_create_q3(adj, u, v):
    key = (u,v) if u<v else (v,u)
    temp = adj | {key}
    others = [x for x in range(16) if x!=u and x!=v]
    for six in combinations(others, 6):
        verts = tuple(sorted([u,v]+list(six)))
        if _subset_has_q3(verts, temp):
            return True
    return False

def _greedy_add(adj):
    n=16
    adj = set(adj)
    changed=True
    while changed and _time_left()>5:
        changed=False
        for i in range(n):
            for j in range(i+1,n):
                if (i,j) in adj:
                    continue
                if not _would_create_q3(adj, i, j):
                    adj.add((i,j))
                    changed=True
                    if _time_left()<5:
                        return adj
    return adj

def construct():
    global _START
    _START = time.time()
    random.seed(12345)
    n=16
    best = None
    best_m = -1
    
    # base K3,13
    base = set()
    for a in range(3):
        for b in range(3,16):
            base.add((a,b))
    cand = _greedy_add(base)
    if len(cand) > best_m:
        best_m = len(cand); best = cand
    
    # random restarts: random dense graph, then greedy add from there won't help (already dense).
    # Instead: random moderate graph then greedy add.
    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    while _time_left() > 8:
        p = random.uniform(0.3, 0.6)
        start = set()
        for (i,j) in all_pairs:
            if random.random() < p:
                start.add((i,j))
        cand = _greedy_add(start)
        if len(cand) > best_m:
            best_m = len(cand); best = cand
    
    # Also try: start full, remove edges that are in Q3s (min removals)
    if _time_left() > 10:
        full = set(all_pairs)
        # repeatedly find a Q3 and remove one of its edges
        removed = 0
        adj = set(full)
        # find any Q3
        def find_q3(adj):
            for verts in combinations(range(16),8):
                if _subset_has_q3(verts, adj):
                    return verts
            return None
        while _time_left() > 6:
            qv = find_q3(adj)
            if qv is None:
                break
            # remove a random edge among those in the q3's cube edges under some embedding
            # we just remove a random edge of the induced subgraph on qv
            edges_in = []
            for a in qv:
                for b in qv:
                    if a<b and (a,b) in adj:
                        edges_in.append((a,b))
            if edges_in:
                e = random.choice(edges_in)
                adj.remove(e)
                removed+=1
        if len(adj) > best_m:
            best_m = len(adj); best = adj
    
    if best is None:
        best = base
    return [[u,v] for (u,v) in best if u<v]
