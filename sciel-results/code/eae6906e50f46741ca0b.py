"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random

_START = time.time()
_DEADLINE = 115.0


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _exact_alpha(adj_bits, n):
    """
    Exact maximum independent set size using branch and bound.
    """
    full_mask = (1 << n) - 1
    comp_bits = [full_mask ^ adj_bits[i] ^ (1 << i) for i in range(n)]
    
    best = [0]
    
    def _color_bound(cand_mask):
        if cand_mask == 0:
            return 0
        verts = []
        m = cand_mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        verts.sort(key=lambda v: bin(comp_bits[v] & cand_mask).count('1'), reverse=True)
        
        colors = {}
        max_c = 0
        for v in verts:
            used = 0
            cv = comp_bits[v] & cand_mask
            cm = cv
            while cm:
                u = (cm & -cm).bit_length() - 1
                if u in colors:
                    used |= (1 << colors[u])
                cm &= cm - 1
            c = 0
            while used & (1 << c):
                c += 1
            colors[v] = c
            if c > max_c:
                max_c = c
        return max_c + 1

    def bnb(cand_mask, cur_size):
        nonlocal best
        if cand_mask == 0:
            if cur_size > best[0]:
                best[0] = cur_size
            return
        
        cb = _color_bound(cand_mask)
        if cur_size + cb <= best[0]:
            return
        
        if cur_size + bin(cand_mask).count('1') <= best[0]:
            return

        verts = []
        m = cand_mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        
        v = max(verts, key=lambda u: bin(adj_bits[u] & cand_mask).count('1'))
        bit_v = 1 << v
        rest = cand_mask & ~bit_v

        nbrs = adj_bits[v] & cand_mask
        new_cand = rest & ~nbrs
        bnb(new_cand, cur_size + 1)

        bnb(rest, cur_size)

    bnb(full_mask, 0)
    return best[0]


def _local_search(n, time_limit=10.0):
    """
    Local search to find a triangle-free graph on n vertices with alpha < 10.
    """
    if _remaining_time() < 2:
        return None
    
    best_adj = None
    best_alpha = float('inf')
    
    while _remaining_time() > 2:
        rng = random.Random()
        adj_bits = [0] * n
        
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if _remaining_time() < 2:
                break
            if adj_bits[u] & adj_bits[v]:
                continue
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
        alpha = _exact_alpha(adj_bits, n)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj_bits[:]
            if alpha < 10:
                return best_adj
            
        steps_since_improve = 0
        local_best_alpha = alpha
        
        while _remaining_time() > 2 and steps_since_improve < 2000:
            u = rng.randint(0, n-1)
            v = rng.randint(0, n-1)
            if u == v:
                continue
            if (adj_bits[u] >> v) & 1:
                continue
            
            if adj_bits[u] & adj_bits[v]:
                continue
            
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
            new_alpha = _exact_alpha(adj_bits, n)
            
            if new_alpha < local_best_alpha:
                local_best_alpha = new_alpha
                steps_since_improve = 0
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_adj = adj_bits[:]
                    if new_alpha < 10:
                        return best_adj
            else:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                steps_since_improve += 1
                
    return best_adj if best_alpha < 10 else None


def _bitset_to_edges(adj_bits, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges


def construct():
    for n_target in range(40, 20, -1):
        if _remaining_time() < 5:
            break
        time_alloc = min(5.0, _remaining_time() - 2.0)
        result_adj = _local_search(n_target, time_limit=time_alloc)
        if result_adj is not None:
            alpha = _exact_alpha(result_adj, n_target)
            if alpha < 10:
                return {'n': n_target, 'edges': _bitset_to_edges(result_adj, n_target)}
    
    # Fallback: n=1
    return {'n': 1, 'edges': []}
