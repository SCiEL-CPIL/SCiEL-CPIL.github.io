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
    # Simple B&B for MIS on N=18
    best = [0]
    
    def _solve(mask, size):
        if not mask:
            if size > best[0]:
                best[0] = size
            return
        # Prune
        if size + mask.bit_count() <= best[0]:
            return
            
        # Pick a vertex with highest degree in remaining graph
        best_v = -1
        best_deg = -1
        c = mask
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            deg = (t_adj[v] & mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        
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

def _local_search(adj, t_adj, score, max_iters=100):
    """Try edge swaps to improve the score"""
    for _ in range(max_iters):
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Try adding a random non-edge
        non_edges = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if not ((adj[u] >> v) & 1):
                    non_edges.append((u, v))
        
        if not non_edges:
            break
            
        u, v = random.choice(non_edges)
        
        # Check if adding edge (u,v) keeps K4-free
        if _is_k4_free_addition(adj, u, v):
            # Add the edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
            # Recompute triangle adjacency
            new_t_adj = _build_triangle_adj(adj)
            new_score = _max_independent_set_size(new_t_adj)
            
            if new_score < score:
                score = new_score
                t_adj = new_t_adj
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        # Try removing a random edge
        edges = [(u, v) for u in range(_N) for v in range(u + 1, _N) if (adj[u] >> v) & 1]
        if edges:
            u, v = random.choice(edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            new_t_adj = _build_triangle_adj(adj)
            new_score = _max_independent_set_size(new_t_adj)
            
            if new_score < score:
                score = new_score
                t_adj = new_t_adj
            else:
                # Revert
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                
    return adj, t_adj, score

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
        
        # Local search to improve
        adj, t_adj, score = _local_search(adj, t_adj, score, max_iters=50)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            # If we find 10 or lower, we can stop or keep looking for 9
            if best_score <= 10:
                # Try a few more to see if we can get lower, but 10 is likely optimal
                pass
        
        iterations += 1
        if iterations > 3000: break # Safety
        
    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
