import time
import random
from collections import deque

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        color = [-1] * n
        
        def backtrack(i):
            if i == n:
                return True
            
            mask = adj[i]
            mask_lower = mask & ((1 << i) - 1)
            
            used_colors = 0
            m = mask_lower
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if color[j] != -1:
                    used_colors |= (1 << color[j])
                m ^= lsb
            
            for c in range(3):
                if not (used_colors & (1 << c)):
                    color[i] = c
                    if backtrack(i + 1):
                        return True
            color[i] = -1
            return False
        
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges

    def make_grotzsch_adj():
        adj = [0] * n
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u, v in cycle:
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    def greedy_expand_deterministic(adj):
        """Repeatedly add edges that don't create triangles, preferring edges
        that block fewer other potential edges (i.e., have fewer common neighbors
        with other non-edges). Multiple passes until stable."""
        improved = True
        while improved:
            improved = False
            # Build list of candidate edges
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        candidates.append((u, v))
            
            # Score each candidate: number of other candidates it would block
            # (i.e., for candidate (u,v), count candidates (a,b) where 
            # adding (u,v) makes (a,b) impossible because a,b share a neighbor)
            # Simpler: just prefer edges with lower degree sum
            scored = []
            for u, v in candidates:
                # If we add (u,v), it blocks any edge (a,b) where a is neighbor of u or v
                # and b is neighbor of u or v, and a!=b, and (a,b) is currently non-edge
                # Actually simpler: just use degree as proxy
                deg_sum = bin(adj[u]).count('1') + bin(adj[v]).count('1')
                scored.append((deg_sum, u, v))
            
            scored.sort()
            
            for deg_sum, u, v in scored:
                if time.time() > deadline:
                    break
                if is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
            
            if time.time() > deadline:
                break
        
        return adj

    def local_search(adj, max_iter=200):
        """Try removing one edge and re-expanding to see if we gain edges."""
        current_count = sum(bin(a).count('1') for a in adj) // 2
        best_adj = list(adj)
        best_count = current_count
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Get current edges
            edges = []
            for u in range(n):
                mask = adj[u]
                mask_higher = mask & ~((1 << (u + 1)) - 1)
                m = mask_higher
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    edges.append((u, v))
                    m ^= lsb
            
            if not edges:
                break
            
            # Try removing a random edge
            u, v = random.choice(edges)
            
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Re-expand
            greedy_expand_deterministic(adj)
            
            new_count = sum(bin(a).count('1') for a in adj) // 2
            
            if new_count > best_count:
                best_count = new_count
                best_adj = list(adj)
                # Keep the improved adj
            else:
                # Restore: need to re-add the edge and re-expand
                # Actually, let's just restart from best_adj
                adj = list(best_adj)
        
        return best_adj, best_count

    def try_random_seed():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.4:
                    if is_triangle_free_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        if not is_3_colorable(adj):
            greedy_expand_deterministic(adj)
            adj, count = local_search(adj, max_iter=50)
            return adj, count
        return None, 0

    # Main loop
    while time.time() < deadline:
        if random.random() < 0.6:
            # Grötzsch seed
            adj = make_grotzsch_adj()
            greedy_expand_deterministic(adj)
            adj, count = local_search(adj, max_iter=100)
            if count > best_count:
                best_count = count
                best_edges = get_edges_from_adj(adj)
        else:
            adj, count = try_random_seed()
            if adj is not None and count > best_count:
                best_count = count
                best_edges = get_edges_from_adj(adj)
        
        # Also try some random restarts with different densities
        if time.time() < deadline:
            for density in [0.3, 0.45, 0.55]:
                if time.time() > deadline:
                    break
                adj = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < density:
                            if is_triangle_free_add(adj, u, v):
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                
                if not is_3_colorable(adj):
                    greedy_expand_deterministic(adj)
                    adj, count = local_search(adj, max_iter=30)
                    if count > best_count:
                        best_count = count
                        best_edges = get_edges_from_adj(adj)

    return best_edges
