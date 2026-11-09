import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 145.0  # seconds, safety margin before 150s limit

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    best = [0]
    
    def popcount(x):
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count('1')
    
    # Order vertices by degree descending to improve pruning
    order = list(range(n))
    order.sort(key=lambda v: popcount(adj[v]), reverse=True)
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE:
            return
        
        if idx == n:
            sz = popcount(current_mask)
            if sz > best[0]:
                best[0] = sz
            return
        
        # Upper bound: current + remaining available
        remaining = 0
        for i in range(idx, n):
            v = order[i]
            if not (forbidden_mask >> v) & 1:
                remaining += 1
        
        if popcount(current_mask) + remaining <= best[0]:
            return
        
        v = order[idx]
        bit = 1 << v
        
        # Option 1: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)
        
        # Option 2: Include v (if not forbidden)
        if not (forbidden_mask & bit):
            new_forbidden = forbidden_mask
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    new_forbidden |= (adj[v] & adj[u])
            
            bb(idx + 1, current_mask | bit, new_forbidden)
    
    bb(0, 0, 0)
    return best[0]

def _try_flip_edge(adj, u, v, n=18):
    """Try flipping edge (u,v). Return new adj if K4-free, else None."""
    has_edge = (adj[u] >> v) & 1
    new_adj = adj[:]
    if has_edge:
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
    else:
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
    
    if not _is_k4_free(new_adj, n):
        return None
    return new_adj

def _local_search(adj, n=18, time_limit=10.0):
    """Local search: try all single edge flips, accept improving moves."""
    best_adj = adj[:]
    best_score = _max_triangle_free_induced(best_adj, n)
    
    if best_score <= 8:
        return best_adj, best_score
    
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if (time.time() - start) > time_limit:
                    break
                new_adj = _try_flip_edge(best_adj, u, v, n)
                if new_adj is None:
                    continue
                new_score = _max_triangle_free_induced(new_adj, n)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj
                    improved = True
                    if best_score <= 8:
                        return best_adj, best_score
    return best_adj, best_score

def _random_k4_free_graph(n=18, p=0.4, seed=None):
    """Generate a random K4-free graph."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Remove edges to make K4-free
    changed = True
    max_iter = 2000
    while changed and max_iter > 0:
        changed = False
        max_iter -= 1
        if _is_k4_free(adj, n):
            break
        # Find a K4 and remove an edge from it
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                common = adj[a] & adj[b]
                c = common
                while c:
                    lsb = c & -c
                    x = lsb.bit_length() - 1
                    c -= lsb
                    if (adj[x] & common) != 0:
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return adj

def construct():
    n = 18
    
    # Start with K6,6,6 (complete 3-partite)
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    best_adj = adj[:]
    best_score = _max_triangle_free_induced(best_adj, n)
    
    # Local search from initial graph
    best_adj, best_score = _local_search(best_adj, n, time_limit=15.0)
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(20):
        if _time_left() < 10:
            break
        
        p = 0.3 + 0.02 * restart
        rand_adj = _random_k4_free_graph(n, p=p, seed=42 + restart)
        
        remaining = min(10.0, _time_left() - 5.0)
        if remaining <= 0:
            break
        rand_adj, rand_score = _local_search(rand_adj, n, time_limit=remaining)
        
        if rand_score < best_score:
            best_score = rand_score
            best_adj = rand_adj
        
        if best_score <= 8:
            break
    
    # Final polish
    if _time_left() > 5:
        best_adj, best_score = _local_search(best_adj, n, time_limit=min(5.0, _time_left() - 1.0))
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
