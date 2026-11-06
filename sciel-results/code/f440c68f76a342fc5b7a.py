import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        cu = (adj[u] & adj[v]).bit_count()
        if cu > 2:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:
                if (adj[u] & adj[w]).bit_count() > 1:
                    return False
            if (adj[u] >> w) & 1:
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    def greedy_build(seed=None):
        """Greedy random construction."""
        rng = random.Random(seed)
        adj = [0] * n
        shuffled = list(all_edges)
        rng.shuffle(shuffled)
        for (u, v) in shuffled:
            if is_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def structured_initial(seed=None):
        """Structured initial graph using circulant base + greedy fill."""
        rng = random.Random(seed)
        adj = [0] * n
        # Try different circulant offsets
        offsets = [1, 2, 3, 4, 5, 7, 8, 9]
        rng.shuffle(offsets)
        for d in offsets[:3]:
            j = (0 + d) % n
            if j != 0:
                if is_k23_free(adj, 0, j):
                    adj[0] |= (1 << j)
                    adj[j] |= (1 << 0)
        # Fill greedily
        shuffled = list(all_edges)
        rng.shuffle(shuffled)
        for (u, v) in shuffled:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng, time_budget=5.0):
        """Local search with 1-swap and 2-swap moves."""
        adj = list(adj)
        search_start = time.time()
        
        # Build edge sets
        present = set()
        missing = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    present.add((u, v))
                else:
                    missing.add((u, v))
        
        best_adj = list(adj)
        best_count = edge_count(adj)
        
        improved = True
        while improved:
            if time.time() - search_start > time_budget:
                break
            improved = False
            
            # Try adding any missing edge
            for (u, v) in list(missing):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    present.add((u, v))
                    missing.discard((u, v))
                    improved = True
                    break
            if improved:
                cnt = edge_count(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = list(adj)
                continue
            
            # Try 1-swap
            found_swap = False
            if present and missing:
                present_list = list(present)
                missing_list = list(missing)
                for _ in range(300):
                    if time.time() - search_start > time_budget:
                        break
                    a, b = rng.choice(present_list)
                    c, d = rng.choice(missing_list)
                    
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    
                    if is_k23_free(adj, c, d):
                        adj[c] |= (1 << d)
                        adj[d] |= (1 << c)
                        present.discard((a, b))
                        present.add((c, d))
                        missing.add((a, b))
                        missing.discard((c, d))
                        improved = True
                        found_swap = True
                        break
                    else:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
            
            if found_swap:
                cnt = edge_count(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = list(adj)
                continue
            
            # Try 2-swap
            found_2swap = False
            if len(present) >= 2 and len(missing) >= 2:
                present_list = list(present)
                missing_list = list(missing)
                for _ in range(100):
                    if time.time() - search_start > time_budget:
                        break
                    (a, b), (c, d) = rng.sample(present_list, 2)
                    (e, f), (g, h) = rng.sample(missing_list, 2)
                    
                    adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                    adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                    
                    ok1 = is_k23_free(adj, e, f)
                    if ok1:
                        adj[e] |= (1 << f); adj[f] |= (1 << e)
                        ok2 = is_k23_free(adj, g, h)
                        if ok2:
                            adj[g] |= (1 << h); adj[h] |= (1 << g)
                            present.discard((a, b))
                            present.discard((c, d))
                            present.add((e, f))
                            present.add((g, h))
                            missing.add((a, b))
                            missing.add((c, d))
                            missing.discard((e, f))
                            missing.discard((g, h))
                            improved = True
                            found_2swap = True
                            break
                        else:
                            adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                    
                    adj[a] |= (1 << b); adj[b] |= (1 << a)
                    adj[c] |= (1 << d); adj[d] |= (1 << c)
            
            if found_2swap:
                cnt = edge_count(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = list(adj)
                continue
            
            if not found_swap and not found_2swap:
                break
        
        return best_adj

    def final_polish(adj, rng):
        """High-precision polish: try all possible single edge additions and swaps."""
        adj = list(adj)
        search_start = time.time()
        
        present = set()
        missing = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    present.add((u, v))
                else:
                    missing.add((u, v))
        
        best_adj = list(adj)
        best_count = edge_count(adj)
        
        # Exhaustive add check
        for (u, v) in list(missing):
            if time.time() - search_start > 10.0:
                break
            if is_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                present.add((u, v))
                missing.discard((u, v))
        
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
        
        # Systematic 1-swap: try all present-missing pairs
        present_list = list(present)
        missing_list = list(missing)
        for a, b in present_list:
            if time.time() - search_start > 10.0:
                break
            for c, d in missing_list:
                if time.time() - search_start > 10.0:
                    break
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                if is_k23_free(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    present.discard((a, b))
                    present.add((c, d))
                    missing.add((a, b))
                    missing.discard((c, d))
                    cnt = edge_count(adj)
                    if cnt > best_count:
                        best_count = cnt
                        best_adj = list(adj)
                        # Continue searching from improved state
                        present_list = list(present)
                        missing_list = list(missing)
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        
        return best_adj

    best_adj = None
    best_count = 0

    # Generate diverse initial solutions
    inits = []
    # Structured inits
    for s in range(5):
        inits.append(structured_initial(s))
    # Greedy inits
    for s in range(10):
        inits.append(greedy_build(s))
    
    # Evaluate inits and run local search on promising ones
    rng_main = random.Random(42)
    for adj in inits:
        if time.time() > deadline - 20.0:
            break
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
        
        # Run local search on this init
        if time.time() < deadline - 15.0:
            improved_adj = local_search(adj, rng_main, time_budget=3.0)
            cnt = edge_count(improved_adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(improved_adj)
    
    # Run more local search iterations on the best solution
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            improved_adj = local_search(best_adj, rng_main, time_budget=2.0)
            cnt = edge_count(improved_adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(improved_adj)
    
    # Final polish
    if time.time() < deadline - 2.0:
        polished = final_polish(best_adj, rng_main)
        cnt = edge_count(polished)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(polished)
    
    return adj_to_edges(best_adj)
