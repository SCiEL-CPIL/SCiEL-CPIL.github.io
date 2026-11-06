import random
import time
import sys

def _popcount(x):
    return bin(x).count('1')

def _has_k4(adj):
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[k] & common:
                        return True
    return False

def _find_k4(adj):
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[w] & common:
                        cw = adj[w] & common
                        if cw:
                            lsb2 = cw & -cw
                            x = lsb2.bit_length() - 1
                            return (u, v, w, x)
    return None

def _has_indep6_complete(adj):
    n = len(adj)
    mask = (1 << n) - 1
    
    def dfs(candidates, size):
        if size == 0:
            return True
        if candidates.bit_count() < size:
            return False
        
        # Branch and bound: try vertices
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            # Neighbors in complement
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            if dfs(new_cands, size - 1):
                return True
        return False

    return dfs(mask, 6)

def _find_indep6_complete(adj):
    n = len(adj)
    mask = (1 << n) - 1
    
    def dfs(candidates, size):
        if size == 0:
            return []
        if candidates.bit_count() < size:
            return None
        
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            res = dfs(new_cands, size - 1)
            if res is not None:
                return [v] + res
        return None

    return dfs(mask, 6)

def _is_valid(adj):
    return not _has_k4(adj) and not _has_indep6_complete(adj)

def _random_graph(n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_search(adj, time_limit):
    n = len(adj)
    start_time = time.time()
    
    # Initial check
    if _is_valid(adj):
        return True

    while time.time() - start_time < time_limit:
        # Find a violation
        k4 = _find_k4(adj)
        if k4:
            u, v, w, x = k4
            edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            ru, rv = random.choice(edges)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            continue
        
        indep = _find_indep6_complete(adj)
        if indep:
            i, j = random.sample(indep, 2)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            continue
            
        # If no violation found by finders, it should be valid
        # But finders are complete, so if they return None, it is valid
        if _is_valid(adj):
            return True

    return _is_valid(adj)

def construct():
    deadline = time.time() + 110
    best_n = 0
    best_adj = None
    
    # Try to find a graph for n=35 down to 20
    for n in range(35, 19, -1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random restarts
        # We use a smarter initialization: start with a random graph, 
        # but maybe bias towards a known structure?
        # For now, random with p=0.5 is okay, but we need more time per attempt.
        
        attempts = 0
        max_attempts = 20
        while time.time() < deadline - 5:
            attempts += 1
            adj = _random_graph(n, 0.5)
            
            # Give each attempt a fair chunk of time, but not too much
            # If n is large, search is harder.
            time_per_attempt = min(2.0, (deadline - time.time() - 5) / max(1, max_attempts - attempts))
            
            if _local_search(adj, time_per_attempt):
                best_n = n
                best_adj = adj
                break
            if attempts >= max_attempts:
                break
    
    if best_n == 0:
        # Fallback to a known valid graph
        # Paley 17 is valid?
        # Let's verify Paley 17 in code or just use a small safe one.
        # A cycle C5 is K4-free and alpha=2.
        # Let's construct a known valid graph for n=17 (Paley)
        # Or just return a small valid graph to ensure non-zero score.
        # We can construct a graph with n=10 easily.
        # But let's try to return the best found.
        # If none found, return a trivial valid graph.
        # E.g., n=5, complete graph K5? No, K5 has K4.
        # n=4, C4? K4-free, alpha=2. Valid.
        # Let's return a random valid graph found in a quick search for small n.
        
        # Quick search for n=20..10
        for n in range(20, 9, -1):
            if time.time() > deadline - 1:
                break
            for _ in range(10):
                adj = _random_graph(n, 0.5)
                if _local_search(adj, 0.5):
                    best_n = n
                    best_adj = adj
                    break
            if best_n > 0:
                break

    if best_n == 0:
        # Absolute fallback: n=4, C4
        best_n = 4
        best_adj = [0, 2, 1, 2] # 0-1, 1-2, 2-3, 3-0
        # adj[0] = 2 (bit 1) -> 0-1
        # adj[1] = 1 (bit 0) | 4 (bit 2) -> 1-0, 1-2. Wait.
        # C4: 0-1, 1-2, 2-3, 3-0
        # adj[0] = 1 | 8 = 9
        # adj[1] = 1 | 2 = 3
        # adj[2] = 2 | 4 = 6
        # adj[3] = 4 | 8 = 12
        best_adj = [9, 3, 6, 12]

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
