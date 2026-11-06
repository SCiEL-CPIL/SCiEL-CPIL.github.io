import time
import random

_N = 16
_M = 65

def _make_adj_deg(edges):
    adj = [0]*_N
    deg = [0]*_N
    for u,v in edges:
        adj[u] |= (1<<v)
        adj[v] |= (1<<u)
        deg[u]+=1
        deg[v]+=1
    return adj, deg

def _score_adj(adj, deg):
    best = 0
    for u in range(_N):
        nu = adj[u]; du = deg[u]
        m = nu
        while m:
            lsb = m & (-m); v = lsb.bit_length()-1; m ^= lsb
            if v <= u: continue
            common = adj[u] & adj[v]
            if not common: continue
            cm = common; maxd = -1
            while cm:
                l2 = cm & (-cm); w = l2.bit_length()-1; cm ^= l2
                if deg[w] > maxd: maxd = deg[w]
            s = du + deg[v] + maxd
            if s > best: best = s
    return best

def _score(edges):
    adj, deg = _make_adj_deg(edges)
    return _score_adj(adj, deg)

def construct():
    start = time.time()
    deadline = start + 117.0
    rng = random.Random(12345)

    all_pairs = [(i,j) for i in range(_N) for j in range(i+1,_N)]

    def seed_for(a):
        b = _N - a
        need = _M - a*b
        if need < 0: return None
        A = list(range(a)); B = list(range(a,_N))
        core = [(u,v) for u in A for v in B]
        internal = []
        # prefer matching on larger part
        big = B if len(B)>=len(A) else A
        small = A if big is B else B
        lst = big[:]; rng.shuffle(lst)
        for i in range(0,len(lst)-1,2):
            if len(internal)>=need: break
            internal.append((lst[i],lst[i+1]))
        if len(internal)<need:
            lst2 = small[:]; rng.shuffle(lst2)
            for i in range(0,len(lst2)-1,2):
                if len(internal)>=need: break
                internal.append((lst2[i],lst2[i+1]))
        if len(internal)<need:
            # add remaining arbitrary internal
            for u in range(_N):
                if len(internal)>=need: break
                for v in range(u+1,_N):
                    if len(internal)>=need: break
                    if (u,v) not in core and (u,v) not in internal and (v,u) not in core:
                        internal.append((u,v))
        return core + internal

    best_edges = None
    best_score = 10**9

    def consider(edges):
        nonlocal best_edges, best_score
        s = _score(edges)
        if s < best_score:
            best_score = s
            best_edges = list(edges)
        return s

    # seeds
    for a in range(3,10):
        e = seed_for(a)
        if e is not None and len(e)==_M:
            consider(e)

    def local_search(edges, max_iters, deadline):
        edge_set = set((min(u,v),max(u,v)) for u,v in edges)
        adj, deg = _make_adj_deg(list(edge_set))
        cur = _score_adj(adj, deg)
        best = cur
        best_set = set(edge_set)
        for it in range(max_iters):
            if time.time() > deadline: break
            best_move = None
            best_ms = cur
            elist = list(edge_set)
            for e in elist:
                for _ in range(3):
                    ne = all_pairs[rng.randrange(len(all_pairs))]
                    if ne in edge_set: continue
                    ns = edge_set - {e} | {ne}
                    a2,d2 = _make_adj_deg(list(ns))
                    s = _score_adj(a2,d2)
                    if s < best_ms:
                        best_ms = s; best_move = (e,ne,s)
            if best_move is None: break
            e,ne,s = best_move
            edge_set = edge_set - {e} | {ne}
            adj,deg = _make_adj_deg(list(edge_set))
            cur = s
            if cur < best:
                best = cur; best_set = set(edge_set)
        return best, best_set

    # local search from best seed and a few others
    for a in range(3,10):
        if time.time() > deadline: break
        e = seed_for(a)
        if e is None or len(e)!=_M: continue
        s, es = local_search(e, 2000, deadline)
        consider(es)

    # iterated local search from best
    if best_edges is not None:
        for _ in range(200):
            if time.time() > deadline: break
            # perturb
            pert = list(best_edges)
            for _ in range(3):
                u = rng.randrange(_N); v = rng.randrange(_N)
                if u==v: continue
                e = (min(u,v),max(u,v))
                if e in pert:
                    pert.remove(e)
                else:
                    # remove a random edge
                    pert.pop(rng.randrange(len(pert)))
                    pert.append(e)
            if len(pert)!=_M: continue
            s, es = local_search(pert, 1500, deadline)
            consider(es)

    return {'n': _N, 'edges': [[u,v] for u,v in best_edges]}
