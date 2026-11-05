import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q): points [x:y:z], adjacency if x1*x2+y1*y2+z1*z2 == 0
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    assert len(points) == q*q + q + 1

    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))

    m = len(edges)

    def edge_key(e):
        return (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])

    def count_c4(u, v):
        # number of 4-cycles u-a-b-v where a~u, b~v, a~b, a!=b
        nu = adj[u]
        nv = adj[v]
        c = 0
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    c += 1
        return c

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    def total_c4():
        # Count C4s using adjacency matrix method
        A = np.zeros((n, n), dtype=float)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        A2 = A @ A
        A4 = A2 @ A2
        trace_A4 = np.trace(A4)
        deg = np.sum(A, axis=1)
        sum_d2 = np.sum(deg**2)
        c4_count = (trace_A4 - 4*sum_d2 + 2*len(edges)) / 8
        return int(round(c4_count))

    # ---- Adjust to exactly 91 edges using local-search-quality moves ----
    if m > 91:
        to_remove = m - 91
        for _ in range(to_remove):
            best = None
            best_c = None
            for (u, v) in list(edges):
                c = count_c4(u, v)
                if best_c is None or c < best_c:
                    best_c = c
                    best = (u, v)
            edges.remove(best)
            remove_edge(best[0], best[1])
    elif m < 91:
        to_add = 91 - m
        edge_set = set(edge_key(e) for e in edges)
        for _ in range(to_add):
            best = None
            best_c = None
            for u in range(n):
                for v in range(u+1, n):
                    k = (u, v)
                    if k in edge_set:
                        continue
                    c = count_c4(u, v)
                    if best_c is None or c < best_c:
                        best_c = c
                        best = (u, v)
            edges.append(best)
            edge_set.add(edge_key(best))
            add_edge(best[0], best[1])

    edge_set = set(edge_key(e) for e in edges)

    def exact_swap_delta(ru, rv, au, av):
        """Compute exact change in total C4 count for removing (ru,rv) and adding (au,av)."""
        c4s_using_ru_rv = 0
        for x in adj[ru]:
            if x == rv:
                continue
            for y in adj[rv]:
                if y == ru or y == x:
                    continue
                if y in adj[x]:
                    c4s_using_ru_rv += 1
        
        c4s_created_au_av = 0
        for x in adj[au]:
            if x == av:
                continue
            for y in adj[av]:
                if y == au or y == x:
                    continue
                if y in adj[x]:
                    c4s_created_au_av += 1
        
        return c4s_created_au_av - c4s_using_ru_rv

    # ---- Phase 1: Intensive local search with steepest descent ----
    best_c4 = total_c4()
    best_edges = list(edges)
    best_adj = [set(s) for s in adj]

    for it in range(200000):
        if time.time() > deadline - 30:
            break
        
        # Try all possible swaps (remove one edge, add one non-edge)
        best_d = 0
        best_move = None
        
        edge_list = list(edges)
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                k = (u, v)
                if k not in edge_set:
                    non_edges.append(k)
        
        # Limit to top candidates for speed
        edge_c4 = {}
        for e in edge_list:
            ek = edge_key(e)
            edge_c4[ek] = count_c4(e[0], e[1])
        
        nonedge_c4 = {}
        for ne in non_edges:
            nonedge_c4[ne] = count_c4(ne[0], ne[1])
        
        rem_sorted = sorted(edge_c4.items(), key=lambda x: x[1])[:30]
        add_sorted = sorted(nonedge_c4.items(), key=lambda x: x[1])[:30]
        
        for rk, rc in rem_sorted:
            ru, rv = rk
            for ak, ac in add_sorted:
                au, av = ak
                if rk == ak:
                    continue
                d = exact_swap_delta(ru, rv, au, av)
                if d < best_d:
                    best_d = d
                    best_move = (rk, ak)
        
        if best_move is not None and best_d < 0:
            rk, ak = best_move
            ru, rv = rk
            au, av = ak
            remove_edge(ru, rv)
            edge_set.discard(rk)
            add_edge(au, av)
            edge_set.add(ak)
            edges.remove((ru, rv))
            edges.append((au, av))
            
            current_c4 = total_c4()
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = list(edges)
                best_adj = [set(s) for s in adj]
        else:
            # No improvement, try random perturbation
            if random.random() < 0.1:
                if edges and non_edges:
                    re = random.choice(edges)
                    re_key = edge_key(re)
                    ae = random.choice(non_edges)
                    remove_edge(re[0], re[1])
                    edge_set.discard(re_key)
                    add_edge(ae[0], ae[1])
                    edge_set.add(ae)
                    edges.remove(re)
                    edges.append(ae)

    # Restore best solution
    adj = [set(s) for s in best_adj]
    edges = list(best_edges)
    edge_set = set(edge_key(e) for e in edges)

    # ---- Phase 2: Simulated Annealing ----
    T = 3.0
    T_min = 0.01
    cooling = 0.9995
    
    current_c4 = total_c4()
    best_c4_sa = current_c4
    best_edges_sa = list(edges)
    best_adj_sa = [set(s) for s in adj]
    
    for it in range(500000):
        if time.time() > deadline:
            break
        if it % 2000 == 0:
            T = max(T_min, T * cooling)
        
        # Random swap
        e_idx = random.randint(0, len(edges)-1)
        ru, rv = edges[e_idx]
        re_key = edge_key((ru, rv))
        
        # Pick random non-edge
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            ae = (u, v) if u < v else (v, u)
            if ae not in edge_set:
                break
        
        d = exact_swap_delta(ru, rv, ae[0], ae[1])
        
        if d < 0 or random.random() < np.exp(-d / T):
            remove_edge(ru, rv)
            edge_set.discard(re_key)
            add_edge(ae[0], ae[1])
            edge_set.add(ae)
            edges[e_idx] = ae
            
            if it % 500 == 0:
                current_c4 = total_c4()
            
            if current_c4 < best_c4_sa:
                best_c4_sa = current_c4
                best_edges_sa = list(edges)
                best_adj_sa = [set(s) for s in adj]
        
        if it % 1000 == 0:
            current_c4 = total_c4()
            if current_c4 < best_c4_sa:
                best_c4_sa = current_c4
                best_edges_sa = list(edges)
                best_adj_sa = [set(s) for s in adj]

    # Use the better of Phase 1 and Phase 2
    if best_c4_sa < best_c4:
        adj = [set(s) for s in best_adj_sa]
        edges = list(best_edges_sa)
        best_c4 = best_c4_sa
    edge_set = set(edge_key(e) for e in edges)

    # ---- Phase 3: Final polish with more iterations ----
    for it in range(100000):
        if time.time() > deadline:
            break
        
        edge_c4 = {}
        for e in edges:
            ek = edge_key(e)
            edge_c4[ek] = count_c4(e[0], e[1])
        
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                k = (u, v)
                if k not in edge_set:
                    non_edges.append(k)
        
        nonedge_c4 = {}
        for ne in non_edges:
            nonedge_c4[ne] = count_c4(ne[0], ne[1])
        
        rem_sorted = sorted(edge_c4.items(), key=lambda x: x[1])[:20]
        add_sorted = sorted(nonedge_c4.items(), key=lambda x: x[1])[:20]
        
        improved = False
        for rk, rc in rem_sorted:
            if improved:
                break
            ru, rv = rk
            for ak, ac in add_sorted:
                au, av = ak
                if rk == ak:
                    continue
                d = exact_swap_delta(ru, rv, au, av)
                if d < 0:
                    remove_edge(ru, rv)
                    edge_set.discard(rk)
                    add_edge(au, av)
                    edge_set.add(ak)
                    edges.remove((ru, rv))
                    edges.append((au, av))
                    improved = True
                    break
        
        if not improved:
            break

    return [[e[0], e[1]] for e in edges]
