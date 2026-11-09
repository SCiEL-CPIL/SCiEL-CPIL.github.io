import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def _random_maximal_tf(n, deadline, rng):
        adj_mask = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                return None
            if adj_mask[u] & adj_mask[v]:
                continue
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    def _has_independent_set_k(adj_mask, n, k, deadline):
        """Return True if there's an independent set of size k."""
        all_mask = (1 << n) - 1
        comp_mask = [all_mask ^ (1 << i) ^ adj_mask[i] for i in range(n)]
        
        comp_deg = [bin(comp_mask[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: comp_deg[i], reverse=True)
        
        new_comp = [0] * n
        pos_map = [0] * n
        for new_idx, old_idx in enumerate(order):
            pos_map[old_idx] = new_idx
            mask = 0
            for old_j in range(n):
                if old_j != old_idx and (comp_mask[old_idx] >> old_j) & 1:
                    mask |= (1 << pos_map[old_j])
            new_comp[new_idx] = mask
        
        def bnb(candidates, need):
            if time.time() > deadline:
                return True
            if need == 0:
                return True
            if candidates == 0:
                return False
            if bin(candidates).count('1') < need:
                return False
            
            c = candidates
            while c:
                lb = c & (-c)
                idx = lb.bit_length() - 1
                c ^= lb
                
                remaining = candidates & (~((1 << (idx + 1)) - 1))
                new_cands = remaining & new_comp[idx]
                
                if bnb(new_cands, need - 1):
                    return True
            return False
        
        return bnb((1 << n) - 1, k)
    
    def _mask_to_edges(adj_mask, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    rng = random.Random(12345)
    
    # Try n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 300 if n <= 32 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            adj_mask = _random_maximal_tf(n, deadline - 20, rng)
            if adj_mask is None:
                break
            
            if not _has_independent_set_k(adj_mask, n, 10, deadline - 15):
                return {'n': n, 'edges': _mask_to_edges(adj_mask, n)}
    
    # Fallback: K_{9,9} on 18 vertices
    n = 18
    edges = []
    for i in range(9):
        for j in range(9, 18):
            edges.append([i, j])
    return {'n': n, 'edges': edges}
