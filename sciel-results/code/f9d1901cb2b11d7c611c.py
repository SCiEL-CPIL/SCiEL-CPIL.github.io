import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def seed_graph():
        adj = [0]*n
        # common[u] is a bitmask: bit w is 1 if u and w have >=2 common neighbors
        common = [0]*n
        for k in range(5):
            for a in range(4):
                for b in range(a+1,4):
                    u = 4*k+a; v = 4*k+b
                    adj[u] |= 1<<v; adj[v] |= 1<<u
                    # update common using bitmasks
                    # For edge (u,v): common[u] gets bits where adj[v] & ~(1<<u) & ~(1<<v)
                    # But we need to track when count reaches 2, not just 1.
                    # Let's use a different approach: common[u][w] = number of common neighbors
                    # Actually, let's use a 2D array but with numpy for speed
                    pass
        # Rebuild common properly
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u]>>v)&1:
                    cu = adj[u] & adj[v]
                    cu &= ~(1<<u) & ~(1<<v)
                    for w in range(n):
                        if (cu>>w)&1:
                            common[u][w]+=1
                            common[v][w]+=1
                            common[w][u]+=1
                            common[w][v]+=1
        return adj, common
    
    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj)//2
    
    def can_add(adj, common, u, v):
        if (adj[u]>>v)&1: return True
        # For adding (u,v): 
        # - For each w in N(u) \ {v}: common[v][w] must be < 2
        # - For each w in N(v) \ {u}: common[u][w] must be < 2
        # - common[u][v] must be < 3
        # Use bitmask: neighbors of u excluding v
        nu = adj[u] & ~(1<<v)
        nv = adj[v] & ~(1<<u)
        # Check: for all w in nu, common[v][w] < 2
        # For all w in nv, common[u][w] < 2
        for w in range(n):
            if w==u or w==v: continue
            if (nu>>w)&1:
                if common[v][w] >= 2: return False
            if (nv>>w)&1:
                if common[u][w] >= 2: return False
        if common[u][v] >= 3: return False
        return True
    
    def do_add(adj, common, u, v):
        adj[u] |= 1<<v; adj[v] |= 1<<u
        for w in range(n):
            if w==u or w==v: continue
            if (adj[v]>>w)&1:
                common[u][w]+=1; common[w][u]+=1
            if (adj[u]>>w)&1:
                common[v][w]+=1; common[w][v]+=1
    
    def do_remove(adj, common, u, v):
        adj[u] &= ~(1<<v); adj[v] &= ~(1<<u)
        for w in range(n):
            if w==u or w==v: continue
            if (adj[v]>>w)&1:
                common[u][w]-=1; common[w][u]-=1
            if (adj[u]>>w)&1:
                common[v][w]-=1; common[w][v]-=1
    
    def local_improve(adj, common):
        changed = True
        while changed:
            changed = False
            for (u,v) in all_edges:
                if time.time() > deadline - 2.0:
                    return adj, common
                if not ((adj[u]>>v)&1) and can_add(adj, common, u, v):
                    do_add(adj, common, u, v)
                    changed = True
        return adj, common
    
    best = None
    best_count = -1
    rng = random.Random(12345)
    
    for attempt in range(5000):
        if time.time() > deadline - 5.0:
            break
        mode = rng.random()
        if mode < 0.3:
            adj, common = seed_graph()
        else:
            adj = [0]*n
            common = [[0]*n for _ in range(n)]
        order = all_edges[:]
        rng.shuffle(order)
        for (u,v) in order:
            if time.time() > deadline - 2.0:
                break
            if not ((adj[u]>>v)&1) and can_add(adj, common, u, v):
                do_add(adj, common, u, v)
        adj, common = local_improve(adj, common)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best = (adj[:], [row[:] for row in common])
        if best is not None and attempt % 10 == 0:
            adj2, common2 = best[0][:], [row[:] for row in best[1]]
            edges_now = [(u,v) for (u,v) in all_edges if (adj2[u]>>v)&1]
            if edges_now:
                u,v = rng.choice(edges_now)
                do_remove(adj2, common2, u, v)
                adj2, common2 = local_improve(adj2, common2)
                cnt2 = edge_count(adj2)
                if cnt2 > best_count:
                    best_count = cnt2
                    best = (adj2[:], [row[:] for row in common2])
    
    if best is None:
        adj, common = seed_graph()
    else:
        adj = best[0]
    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u]>>v)&1:
                res.append([u,v])
    return res
