import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        nu = adj[u]
        nv = adj[v]
        
        # Check case: v is common neighbor of x, y
        # We need |N(u) & N(x) & N(y)| == 2 for some x,y in N(v)
        # Iterate over pairs in N(v)
        nv_list = []
        m = nv
        while m:
            low = m & (-m)
            idx = low.bit_length() - 1
            nv_list.append(idx)
            m ^= low
            
        for i in range(len(nv_list)):
            x = nv_list[i]
            ax = adj[x]
            for j in range(i+1, len(nv_list)):
                y = nv_list[j]
                ay = adj[y]
                common = nu & ax & ay
                if common.bit_count() == 2:
                    return False
        
        # Check case: u is common neighbor of x, y
        nu_list = []
        m = nu
        while m:
            low = m & (-m)
            idx = low.bit_length() - 1
            nu_list.append(idx)
            m ^= low
            
        for i in range(len(nu_list)):
            x = nu_list[i]
            ax = adj[x]
            for j in range(i+1, len(nu_list)):
                y = nu_list[j]
                ay = adj[y]
                common = nv & ax & ay
                if common.bit_count() == 2:
                    return False
        
        return True

    def local_search(adj, max_iters=500):
        adj = adj[:]
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            added = False
            # Try to add edges
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if adj[u] & (1 << v):
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if added:
                continue
            
            # Try swaps
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_in or not edges_out:
                break
                
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            # Limit search space for speed
            for e_rem in edges_in[:20]:
                if swapped: break
                u1, v1 = e_rem
                for e_add in edges_out[:20]:
                    u2, v2 = e_add
                    if (u1, v1) == (u2, v2): continue
                    # Temp remove
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    if can_add_edge(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        swapped = True
                        break
                    else:
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                if not swapped:
                    continue
            
            if not swapped:
                break
        return adj

    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Run multiple trials
    trials = 0
    while time.time() < deadline - 10:
        trials += 1
        random.seed(42 + trials)
        
        # Start with random sparse graph
        adj = [0] * n
        p = 0.3
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Repair if invalid
        for _ in range(50):
            if is_valid(adj):
                break
            # Find violation and remove random edge from it
            found = False
            for i in range(n):
                if found: break
                for j in range(i+1, n):
                    if found: break
                    cij = adj[i] & adj[j]
                    if cij.bit_count() < 3: continue
                    for k in range(j+1, n):
                        cijk = cij & adj[k]
                        if cijk.bit_count() >= 3:
                            # Remove edge incident to i, j, or k
                            verts = [i, j, k]
                            random.shuffle(verts)
                            for v in verts:
                                if adj[v]:
                                    m = adj[v]
                                    low = m & (-m)
                                    idx = low.bit_length() - 1
                                    adj[v] &= ~(1 << idx)
                                    adj[idx] &= ~(1 << v)
                                    found = True
                                    break
                                if found: break
                            if found: break
                    if found: break
                if found: break
        
        adj = local_search(adj)
        update_best(adj)
        
        # Try a few more local searches from the best found so far?
        # Or just restart.

    if not best_edges:
        return [[i, i+1] for i in range(19)]
        
    return best_edges
