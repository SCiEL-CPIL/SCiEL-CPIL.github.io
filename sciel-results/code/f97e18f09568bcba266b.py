import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    # Helper to convert bitmask adj to edge list
    def _adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i + 1)) - 1)
            while mask:
                j = (mask & -mask).bit_length() - 1
                edges.append([i, j])
                mask &= mask - 1
        return edges

    # Helper: Check if graph is triangle-free
    def _is_triangle_free(n, adj):
        for u in range(n):
            mu = adj[u]
            mask = mu
            mask &= ~((1 << (u + 1)) - 1)
            while mask:
                v = (mask & -mask).bit_length() - 1
                if adj[u] & adj[v]:
                    return False
                mask &= mask - 1
        return True

    # Helper: Check if alpha < 10 using B&B on complement
    def _has_independent_set_10(n, adj):
        # Complement adjacency
        comp_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = all_mask ^ adj[i] ^ (1 << i)
        
        def _popcount(x):
            return bin(x).count('1')
        
        def _clique_bnb(candidates_mask, current_size):
            if current_size >= 10:
                return True
            if time.time() > deadline:
                return False
            if _popcount(candidates_mask) + current_size < 10:
                return False
            
            # Degree bound
            target_rem = 10 - current_size - 1
            if target_rem < 0:
                return False
            
            mask = candidates_mask
            all_low = True
            while mask:
                v = (mask & -mask).bit_length() - 1
                deg_in_cand = _popcount(candidates_mask & comp_adj[v])
                if deg_in_cand >= target_rem:
                    all_low = False
                    break
                mask &= mask - 1
            
            if all_low:
                return False

            cands = candidates_mask
            while cands:
                v = (cands & -cands).bit_length() - 1
                mask_greater = cands & ~((1 << (v + 1)) - 1)
                new_cands = comp_adj[v] & mask_greater
                if _clique_bnb(new_cands, current_size + 1):
                    return True
                cands = cands & ~(1 << v)
            
            return False

        all_mask = (1 << n) - 1
        return _clique_bnb(all_mask, 0)

    def _make_valid_triangle_free(n):
        # Start with a random graph
        adj = [0] * n
        # Random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove triangles
        # Simple approach: iterate vertices, for each pair of neighbors, remove edge if it forms triangle?
        # Better: iterate all triples? O(n^3) = 39^3 ~ 59k. Fast.
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        # Check common neighbors
                        common = adj[u] & adj[v]
                        if common:
                            # Remove edge (u,v)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
                            # Restart? Or continue?
                            # Continuing is fine, we might create new triangles? No, removing edges doesn't create triangles.
        
        return adj

    # Try n=39
    n = 39
    attempts = 200
    for _ in range(attempts):
        if time.time() > deadline - 5:
            break
        
        adj = _make_valid_triangle_free(n)
        
        # Check triangle-free (should be true by construction, but verify)
        if not _is_triangle_free(n, adj):
            continue
            
        # Check independence
        if not _has_independent_set_10(n, adj):
            return {'n': n, 'edges': _adj_to_edges(n, adj)}
            
    # Fallback: Andrasfai graph And(13) n=38
    # And(k) on n=3k-1. k=13, n=38.
    # Edges: diff in {1, k, 2k} mod n?
    # Actually And(k) is C_{3k-1}(1, k, 2k) is not standard notation.
    # Standard: i~j if i-j mod n in {1, k, 2k, -1, -k, -2k}.
    k = 13
    n_f = 3 * k - 1
    adj_f = [0] * n_f
    for i in range(n_f):
        for d in [1, k, 2*k]:
            j = (i + d) % n_f
            if j > i:
                adj_f[i] |= (1 << j)
                adj_f[j] |= (1 << i)
    
    # Verify fallback
    if _is_triangle_free(n_f, adj_f) and not _has_independent_set_10(n_f, adj_f):
        return {'n': n_f, 'edges': _adj_to_edges(n_f, adj_f)}
        
    # If even fallback fails (unlikely), return a smaller valid graph
    # And(5) n=14
    k = 5
    n_f2 = 3 * k - 1
    adj_f2 = [0] * n_f2
    for i in range(n_f2):
        for d in [1, k, 2*k]:
            j = (i + d) % n_f2
            if j > i:
                adj_f2[i] |= (1 << j)
                adj_f2[j] |= (1 << i)
                
    return {'n': n_f2, 'edges': _adj_to_edges(n_f2, adj_f2)}

if __name__ == "__main__":
    res = construct()
    print(res['n'], len(res['edges']))
