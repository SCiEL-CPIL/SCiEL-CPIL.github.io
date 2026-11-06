import time
import random

def _popcount(x):
    return bin(x).count('1')

def _ctz(x):
    if x == 0:
        return -1
    return (x & -x).bit_length() - 1

def _has_independent_set_10(n, adj_mask):
    all_mask = (1 << n) - 1
    
    def dfs(candidates, depth, path):
        if depth >= 10:
            return True, path
        if not candidates:
            return False, None
        if depth + _popcount(candidates) < 10:
            return False, None
        
        # Heuristic: pick vertex with max degree in candidates
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            v = _ctz(c)
            c &= c - 1
            deg = _popcount(adj_mask[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        v = best_v
        v_bit = 1 << v
        
        # Include v
        new_cand = candidates & ~adj_mask[v] & ~v_bit
        res, p = dfs(new_cand, depth + 1, path + [v])
        if res:
            return True, p
            
        # Exclude v
        new_cand = candidates & ~v_bit
        res, p = dfs(new_cand, depth, path)
        if res:
            return True, p
            
        return False, None

    return dfs(all_mask, 0, [])

def _find_independent_set_10(n, adj_mask):
    found, path = _has_independent_set_10(n, adj_mask)
    if found:
        return path
    return None

def _is_triangle_free(n, adj_mask):
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_mask[u] >> v) & 1:
                if adj_mask[u] & adj_mask[v]:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Fallback: And(4) on 16 vertices
    # And(4): 4 groups of 4. Edges between G_i and G_{i+1 mod 4}.
    def make_and(k):
        n = k*k
        adj_mask = [0]*n
        edges = []
        for i in range(k):
            for j in range(k):
                u = i*k + j
                # Connect to next and prev group
                for jj in range(k):
                    v_next = ((i+1)%k)*k + jj
                    v_prev = ((i-1)%k)*k + jj
                    if u < v_next:
                        adj_mask[u] |= (1 << v_next)
                        adj_mask[v_next] |= (1 << u)
                        edges.append([u, v_next])
                    if u < v_prev:
                        adj_mask[u] |= (1 << v_prev)
                        adj_mask[v_prev] |= (1 << u)
                        edges.append([u, v_prev])
        return n, edges, adj_mask

    best_n, best_edges, _ = make_and(4)
    
    # Try to find larger graphs
    for target_n in [40, 39, 38, 37, 36, 35, 30, 26]:
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            n = target_n
            adj_mask = [0]*n
            edges = []
            
            # Random triangle-free generation
            random.seed(attempt)
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            p = 0.3
            for u, v in all_edges:
                if random.random() < p:
                    if not (adj_mask[u] & adj_mask[v]):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        edges.append([u, v])
                        
            # Local search
            for step in range(1000):
                if time.time() > deadline:
                    break
                iset = _find_independent_set_10(n, adj_mask)
                if not iset:
                    found = True
                    best_n = n
                    best_edges = edges
                    break
                
                # Try to add edges within iset
                improved = False
                for i in range(len(iset)):
                    for j in range(i+1, len(iset)):
                        u, v = iset[i], iset[j]
                        if not (adj_mask[u] & (1 << v)):
                            # Check triangle
                            if not (adj_mask[u] & adj_mask[v]):
                                # Add edge
                                adj_mask[u] |= (1 << v)
                                adj_mask[v] |= (1 << u)
                                edges.append([u, v])
                                improved = True
                                break
                    if improved:
                        break
                
                if not improved:
                    # Stuck, restart or break
                    break
                    
            if found:
                break
                
    return {'n': best_n, 'edges': best_edges}
