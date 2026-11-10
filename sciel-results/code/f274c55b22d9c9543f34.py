import time
import random

_START = time.time()
_DEADLINE = 140.0
_N = 18
_MASK_ALL = (1 << _N) - 1

def _elapsed():
    return time.time() - _START

def _is_k4_free_addition(adj, u, v):
    if (adj[u] >> v) & 1:
        return True
    common = adj[u] & adj[v]
    # Check if any two vertices in common are adjacent -> would form K4
    c = common
    while c:
        lsb = c & -c
        x = lsb.bit_length() - 1
        c ^= lsb
        if adj[x] & common:
            return False
    return True

def _build_triangle_adj(adj):
    t_adj = [0] * _N
    for u in range(_N):
        for v in range(u + 1, _N):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                if common:
                    t_adj[u] |= 1 << v
                    t_adj[v] |= 1 << u
    return t_adj

def _greedy_mis_lower_bound(t_adj, mask):
    """Greedy lower bound for MIS: repeatedly pick vertex with min degree in remaining."""
    count = 0
    m = mask
    while m:
        # Find vertex with minimum degree in m
        best_v = -1
        best_deg = _N + 1
        c = m
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            deg = (t_adj[v] & m).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
        count += 1
        m &= ~t_adj[best_v] & ~(1 << best_v)
    return count

def _max_independent_set_size(t_adj):
    """Branch and bound for exact MIS on N=18."""
    best = [0]
    
    def _solve(mask, size):
        if not mask:
            if size > best[0]:
                best[0] = size
            return
        # Prune: upper bound
        if size + mask.bit_count() <= best[0]:
            return
        # Greedy lower bound to update best
        lb = size + _greedy_mis_lower_bound(t_adj, mask)
        if lb > best[0]:
            best[0] = lb
            if size + mask.bit_count() <= best[0]:
                return
        
        # Pick vertex with max degree in mask for branching
        v = -1
        max_deg = -1
        c = mask
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            deg = (t_adj[x] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = x
        
        # Branch 1: Include v
        new_mask = mask & ~t_adj[v] & ~(1 << v)
        _solve(new_mask, size + 1)
        
        # Branch 2: Exclude v
        _solve(mask & ~(1 << v), size)

    _solve(_MASK_ALL, 0)
    return best[0]

def _generate_random_k4free():
    adj = [0] * _N
    edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    random.shuffle(edges)
    for u, v in edges:
        if _is_k4_free_addition(adj, u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj

def _local_search_polish(adj, current_score, deadline):
    """Try to improve by adding edges (if K4-free) or removing edges to reduce max triangle-free subgraph."""
    improved = True
    while improved and _elapsed() < deadline - 3:
        improved = False
        t_adj = _build_triangle_adj(adj)
        score = _max_independent_set_size(t_adj)
        
        # Try adding each missing edge
        for u in range(_N):
            for v in range(u + 1, _N):
                if not ((adj[u] >> v) & 1):
                    if _is_k4_free_addition(adj, u, v):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        new_t_adj = _build_triangle_adj(adj)
                        new_score = _max_independent_set_size(new_t_adj)
                        if new_score < score:
                            score = new_score
                            improved = True
                        else:
                            adj[u] ^= 1 << v
                            adj[v] ^= 1 << u
                        if _elapsed() > deadline - 3:
                            return adj, score
        
        # Try removing each existing edge
        for u in range(_N):
            for v in range(u + 1, _N):
                if (adj[u] >> v) & 1:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    new_t_adj = _build_triangle_adj(adj)
                    new_score = _max_independent_set_size(new_t_adj)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                    if _elapsed() > deadline - 3:
                        return adj, score
    
    return adj, score

def construct():
    deadline = _START + _DEADLINE
    best_score = 18
    best_adj = [0] * _N
    
    random.seed(42)
    
    iterations = 0
    while _elapsed() < deadline - 10:
        adj = _generate_random_k4free()
        t_adj = _build_triangle_adj(adj)
        score = _max_independent_set_size(t_adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
        
        iterations += 1
        if iterations > 2000:
            break
    
    # Polish the best solution found
    if best_adj and best_score < 18:
        polished_adj, polished_score = _local_search_polish(best_adj[:], best_score, deadline)
        if polished_score < best_score:
            best_score = polished_score
            best_adj = polished_adj
    
    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
