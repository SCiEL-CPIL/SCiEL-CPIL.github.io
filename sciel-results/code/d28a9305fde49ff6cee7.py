import time
import random

_START = time.time()
_DEADLINE = 120.0
_N = 18
_MASK_ALL = (1 << _N) - 1

def _elapsed():
    return time.time() - _START

def _is_k4_free_addition(adj, u, v):
    if (adj[u] >> v) & 1:
        return True
    common = adj[u] & adj[v]
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

def _max_independent_set_size(t_adj):
    best = [0]
    
    def _solve(mask, size):
        if not mask:
            if size > best[0]:
                best[0] = size
            return
        
        # Prune
        if size + mask.bit_count() <= best[0]:
            return
            
        # Heuristic: Pick vertex with highest degree in the induced subgraph
        # This is O(N^2) per node, but N=18 is small.
        # Or just pick lowest set bit for speed?
        # Let's try picking the vertex with max degree in mask to prune better.
        max_deg = -1
        v = -1
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            m ^= lsb
            deg = (t_adj[x] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = x
        
        if v == -1:
            return

        # Branch 1: Include v
        # Neighbors of v in mask must be removed
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

def construct():
    deadline = _START + _DEADLINE
    best_score = 18
    best_adj = [0] * _N
    
    random.seed(42)
    
    # Try a few random restarts with local search
    while _elapsed() < deadline - 5:
        adj = _generate_random_k4free()
        t_adj = _build_triangle_adj(adj)
        score = _max_independent_set_size(t_adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            if best_score <= 9:
                # 9 is very good, likely optimal or near optimal
                # We can break early if we hit 9 or 8
                if best_score <= 8:
                    break
                # Continue a bit to see if we can get 8
                pass
        
        # Local Search Phase
        # Try to improve the current adj
        improved = True
        while improved and _elapsed() < deadline - 5:
            improved = False
            # Get list of non-edges
            non_edges = []
            for u in range(_N):
                for v in range(u + 1, _N):
                    if not ((adj[u] >> v) & 1):
                        non_edges.append((u, v))
            
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if _elapsed() > deadline - 5:
                    break
                
                # Check if adding (u,v) is valid
                if _is_k4_free_addition(adj, u, v):
                    # Tentatively add
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    
                    # Evaluate new score
                    new_t_adj = _build_triangle_adj(adj)
                    new_score = _max_independent_set_size(new_t_adj)
                    
                    if new_score < score:
                        score = new_score
                        t_adj = new_t_adj
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_adj = adj[:]
                            if best_score <= 8:
                                break
                    else:
                        # Revert
                        adj[u] ^= 1 << v
                        adj[v] ^= 1 << u
            
            if best_score <= 8:
                break

    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
