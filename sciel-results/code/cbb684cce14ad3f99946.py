import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def seed_graph():
        # Start with 5 disjoint K4s (5*6=30 edges)
        adj = [0]*n
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    u = 4*k+a; v = 4*k+b
                    adj[u] |= 1<<v; adj[v] |= 1<<u
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
        nu = adj[u] & ~(1<<v)
        nv = adj[v] & ~(1<<u)
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
    
    def hill_climb(adj, common):
        # Try to swap edges: remove one, add another
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for (u1,v1) in all_edges:
                if time.time() > deadline - 2.0:
                    return adj, common
                if ((adj[u1]>>v1)&1):
                    # Try removing (u1,v1) and adding some other edge
                    do_remove(adj, common, u1, v1)
                    added = False
                    for (u2,v2) in all_edges:
                        if (u2,v2)==(u1,v1) or (u2,v2)==(v1,u1):
                            continue
                        if not ((adj[u2]>>v2)&1) and can_add(adj, common, u2, v2):
                            do_add(adj, common, u2, v2)
                            added = True
                            changed = True
                            break
                    if not added:
                        do_add(adj, common, u1, v1)
        return adj, common
    
    best = None
    best_count = -1
    rng = random.Random(12345)
    
    # Phase 1: Random greedy + local improvement
    for attempt in range(3000):
        if time.time() > deadline - 5.0:
            break
        mode = rng.random()
        if mode < 0.4:
            adj, common = seed_graph()
        elif mode < 0.7:
            adj = [0]*n
            common = [[0]*n for _ in range(n)]
        else:
            # Start from a random sparse graph
            adj = [0]*n
            common = [[0]*n for _ in range(n)]
            for (u,v) in all_edges:
                if rng.random() < 0.1:
                    do_add(adj, common, u, v)
        
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
    
    # Phase 2: Perturbation + local improvement
    if best is not None:
        for attempt in range(2000):
            if time.time() > deadline - 3.0:
                break
            adj2 = best[0][:]
            common2 = [row[:] for row in best[1]]
            
            # Perturb: remove k random edges
            k = rng.randint(1, 5)
            edges_now = [(u,v) for (u,v) in all_edges if (adj2[u]>>v)&1]
            if not edges_now:
                break
            for _ in range(k):
                if not edges_now:
                    break
                u,v = rng.choice(edges_now)
                do_remove(adj2, common2, u, v)
                edges_now = [(u,v) for (u,v) in all_edges if (adj2[u]>>v)&1]
            
            adj2, common2 = local_improve(adj2, common2)
            adj2, common2 = hill_climb(adj2, common2)
            cnt2 = edge_count(adj2)
            if cnt2 > best_count:
                best_count = cnt2
                best = (adj2[:], [row[:] for row in common2])
    
    # Phase 3: More aggressive local search on best
    if best is not None:
        adj = best[0][:]
        common = [row[:] for row in best[1]]
        for _ in range(10):
            if time.time() > deadline - 1.0:
                break
            adj, common = hill_climb(adj, common)
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best = (adj[:], [row[:] for row in common])
    
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
