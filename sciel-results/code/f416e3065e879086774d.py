import time
import random
import copy

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # Check common(u, v) after adding:
        cu = (adj[u] & adj[v]).bit_count()
        if cu > 2:
            return False
        
        # For each w != u, v:
        for w in range(n):
            if w == u or w == v:
                continue
            w_in_v = (adj[v] >> w) & 1
            w_in_u = (adj[u] >> w) & 1
            if w_in_v:
                # v becomes common neighbor of u and w
                common_uw = (adj[u] & adj[w]).bit_count()
                if common_uw > 1:
                    return False
            if w_in_u:
                # u becomes common neighbor of v and w
                common_vw = (adj[v] & adj[w]).bit_count()
                if common_vw > 1:
                    return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def greedy_build(seed=None):
        """Greedy random construction with optional seed for reproducibility."""
        rng = random.Random(seed)
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if is_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, max_iters=2000, rng=None):
        """Local search: try adding edges, then try 1-swap and 2-swap moves."""
        if rng is None:
            rng = random.Random()
        adj = list(adj)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try adding any missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_k23_free(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            
            # Try 1-swap: remove (a,b), add (c,d)
            found_swap = False
            for _ in range(50):  # Try multiple random swap attempts
                a, b = rng.choice([(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1])
                c, d = rng.choice([(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)])
                if (a, b) == (c, d) or (a, b) == (d, c):
                    continue
                
                # Remove (a, b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                # Try adding (c, d)
                if is_k23_free(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    improved = True
                    found_swap = True
                    break
                else:
                    # Restore (a, b)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            
            if found_swap:
                continue
            
            # Try 2-swap: remove (a,b) and (c,d), add (e,f) and (g,h)
            found_2swap = False
            for _ in range(20):
                edges_present = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                edges_missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
                
                if len(edges_present) < 2 or len(edges_missing) < 2:
                    break
                
                (a, b), (c, d) = rng.sample(edges_present, 2)
                (e, f), (g, h) = rng.sample(edges_missing, 2)
                
                # Check no overlap
                if len({(a,b),(c,d),(e,f),(g,h)}) < 4:
                    continue
                
                # Remove (a,b) and (c,d)
                adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                
                # Try adding (e,f) and (g,h)
                if is_k23_free(adj, e, f):
                    adj[e] |= (1 << f); adj[f] |= (1 << e)
                    if is_k23_free(adj, g, h):
                        adj[g] |= (1 << h); adj[h] |= (1 << g)
                        improved = True
                        found_2swap = True
                        break
                    else:
                        # Undo (e,f)
                        adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                
                # Restore (a,b) and (c,d)
                adj[a] |= (1 << b); adj[b] |= (1 << a)
                adj[c] |= (1 << d); adj[d] |= (1 << c)
            
            if not found_2swap and not found_swap:
                break
        
        return adj

    def structured_initial():
        """Create a structured initial graph using block construction."""
        adj = [0] * n
        # Partition into 5 groups of 4
        groups = [[i*4+j for j in range(4)] for i in range(5)]
        
        # Within each group: complete graph
        for g in groups:
            for i in range(len(g)):
                for j in range(i+1, len(g)):
                    u, v = g[i], g[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Between groups: add edges in a pattern that avoids K2,3
        # For each pair of groups, add a matching or sparse connection
        for gi in range(5):
            for gj in range(gi+1, 5):
                # Add a perfect matching between groups gi and gj
                for k in range(4):
                    u = groups[gi][k]
                    v = groups[gj][(k + gi) % 4]  # Shifted matching
                    if is_k23_free(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        return adj

    best_adj = None
    best_count = 0

    # Start with structured initial
    adj = structured_initial()
    cnt = edge_count(adj)
    if cnt > best_count:
        best_adj = list(adj)
        best_count = cnt
    
    # Local search on structured initial
    if time.time() < deadline - 10.0:
        adj_ls = local_search(adj, max_iters=3000)
        cnt_ls = edge_count(adj_ls)
        if cnt_ls > best_count:
            best_adj = list(adj_ls)
            best_count = cnt_ls

    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 5.0:
        seed = restart_count
        adj = greedy_build(seed)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt
        
        # Local search
        if time.time() < deadline - 5.0:
            rng = random.Random(seed + 1000)
            adj_ls = local_search(adj, max_iters=2000, rng=rng)
            cnt_ls = edge_count(adj_ls)
            if cnt_ls > best_count:
                best_adj = list(adj_ls)
                best_count = cnt_ls
        
        restart_count += 1

    if best_adj is None:
        # Fallback: simple construction
        best_adj = greedy_build()

    return adj_to_edges(best_adj)
