import time
import itertools
import random

Q3_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
Q3_ADJ = {i: set() for i in range(8)}
for u, v in Q3_EDGES:
    Q3_ADJ[u].add(v)
    Q3_ADJ[v].add(u)

def contains_q3(vertices, edge_set):
    v_map = {v: i for i, v in enumerate(vertices)}
    local_adj = [set() for _ in range(8)]
    deg = [0]*8
    for u in vertices:
        for v in vertices:
            if u < v and (u, v) in edge_set:
                i, j = v_map[u], v_map[v]
                if i != j:
                    local_adj[i].add(j)
                    local_adj[j].add(i)
                    deg[i] += 1
                    deg[j] += 1
    for d in deg:
        if d < 3:
            return False
    if sum(deg) // 2 < 12:
        return False
    
    used = [False]*8
    mapping = [None]*8
    order = [0, 1, 3, 4, 2, 5, 6, 7]
    
    def bt(step):
        if step == 8:
            return all(mapping[v] in local_adj[mapping[u]] for u, v in Q3_EDGES)
        q3_v = order[step]
        for l_v in range(8):
            if used[l_v]:
                continue
            ok = all(mapping[nb] is None or mapping[nb] in local_adj[l_v] for nb in Q3_ADJ[q3_v])
            if not ok:
                continue
            used[l_v] = True
            mapping[q3_v] = l_v
            if bt(step + 1):
                return True
            used[l_v] = False
            mapping[q3_v] = None
        return False
    return bt(0)

def construct():
    start = time.time()
    # Start with K_{4,12}
    edges = set((a, b) for a in range(4) for b in range(4, 16))
    best = set(edges)
    best_n = len(edges)
    
    all_e = set((u, v) for u in range(16) for v in range(u+1, 16))
    cands = sorted(all_e - edges, key=lambda e: 0 if (e[0] < 4 and e[1] < 4) or (e[0] >= 4 and e[1] >= 4) else 1)
    
    cur = set(edges)
    
    def add_ok(e):
        u, v = e
        new = cur | {e}
        others = [x for x in range(16) if x != u and x != v]
        for c in itertools.combinations(others, 6):
            verts = tuple(sorted(c + (u, v)))
            if contains_q3(verts, new):
                return False
        return True
    
    for e in cands:
        if time.time() - start > 80:
            break
        if add_ok(e):
            cur.add(e)
    
    if len(cur) > best_n:
        best, best_n = set(cur), len(cur)
    
    # Local search
    for _ in range(20):
        if time.time() - start > 110:
            break
        imp = False
        for rem in list(cur):
            if time.time() - start > 110:
                break
            base = cur - {rem}
            for add in cands:
                if add in cur:
                    continue
                if add_ok(add) if False else True:  # Need to check with base
                    # Check if adding 'add' to 'base' is ok
                    u, v = add
                    new = base | {add}
                    others = [x for x in range(16) if x != u and x != v]
                    ok = True
                    for c in itertools.combinations(others, 6):
                        verts = tuple(sorted(c + (u, v)))
                        if contains_q3(verts, new):
                            ok = False
                            break
                    if ok:
                        cur = new
                        imp = True
                        break
            if imp or time.time() - start > 110:
                break
        if not imp:
            break
    
    if len(cur) > best_n:
        best, best_n = set(cur), len(cur)
    
    return [list(e) for e in best]
