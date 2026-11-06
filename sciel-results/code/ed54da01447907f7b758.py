import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    n = 18
    full_mask = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_k4_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if adj[w] & C:
                            return False
                        temp &= temp - 1
        return True
    
    def get_triangles(adj):
        tris = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if w > v:
                            if adj[v] & (1 << w):
                                tris.append((1 << u) | (1 << v) | (1 << w))
                        temp &= temp - 1
        return tris
    
    def max_triangle_free_size(adj, triangles):
        """Find largest triangle-free induced subgraph via branch and bound."""
        if not triangles:
            return n
        
        # Precompute: for each vertex, which triangles contain it
        # Use a recursive branch and bound approach
        # State: (current_size, available_vertices_mask, triangles_to_hit)
        # We want to maximize size while hitting all triangles
        
        # Alternative: try to find if there's a triangle-free set of size k
        # by checking all subsets - but that's what we're trying to avoid.
        
        # Better: use the fact that we need to find a maximum independent set
        # in the 3-uniform hypergraph of triangles.
        # This is NP-hard but n=18 is small enough for B&B.
        
        # B&B: build a set S, prune if any triangle is fully in S
        # Maximize |S|
        
        best = [0]
        tris_list = triangles
        num_tris = len(tris_list)
        
        # For speed, represent triangles as list of 3 vertex indices
        tris_idx = []
        for t in tris_list:
            verts = []
            for i in range(n):
                if t & (1 << i):
                    verts.append(i)
            tris_idx.append(tuple(verts))
        
        # B&B: at each step, decide include or exclude vertex i
        # Prune: if including would complete a triangle, skip
        # Upper bound: current_size + remaining_vertices
        
        def bb(idx, current_mask, current_size):
            if time.time() > deadline - 2:
                return
            if current_size > best[0]:
                best[0] = current_size
            
            if idx == n:
                return
            
            # Upper bound
            if current_size + (n - idx) <= best[0]:
                return
            
            # Option 1: exclude vertex idx
            bb(idx + 1, current_mask, current_size)
            
            # Option 2: include vertex idx (if it doesn't complete any triangle)
            v = idx
            v_bit = 1 << v
            new_mask = current_mask | v_bit
            safe = True
            for (a, b, c) in tris_idx:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)) and (current_mask & (1 << c)):
                    # This triangle is already complete - shouldn't happen if we're careful
                    safe = False
                    break
                # Check if adding v completes this triangle
                tri_mask = (1 << a) | (1 << b) | (1 << c)
                if (new_mask & tri_mask) == tri_mask:
                    safe = False
                    break
            
            if safe:
                bb(idx + 1, new_mask, current_size + 1)
        
        bb(0, 0, 0)
        return best[0]
    
    def can_add_edge_no_k4(adj, u, v):
        if adj[u] & (1 << v):
            return False
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            if adj[w] & C:
                return False
            temp &= temp - 1
        return True
    
    def random_greedy_construction(rng):
        adj = [0] * n
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(possible_edges)
        for u, v in possible_edges:
            if can_add_edge_no_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def local_search(adj, triangles, current_val, rng, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if not (adj[u] & (1 << v)):
                        if can_add_edge_no_k4(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            new_tris = get_triangles(adj)
                            new_val = max_triangle_free_size(adj, new_tris)
                            if new_val < current_val:
                                triangles = new_tris
                                current_val = new_val
                                improved = True
                            else:
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
            # Try removing edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        new_tris = get_triangles(adj)
                        new_val = max_triangle_free_size(adj, new_tris)
                        if new_val < current_val:
                            triangles = new_tris
                            current_val = new_val
                            improved = True
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj, triangles, current_val
    
    best_val = n
    best_adj = None
    best_triangles = None
    
    rng = random.Random(42)
    
    # Try multiple random constructions
    num_constructions = 50
    for i in range(num_constructions):
        if time.time() > deadline - 30:
            break
        
        adj = random_greedy_construction(rng)
        triangles = get_triangles(adj)
        val = max_triangle_free_size(adj, triangles)
        
        if val < best_val:
            best_val = val
            best_adj = adj[:]
            best_triangles = triangles[:]
        
        # Local search on this construction
        remaining = deadline - time.time()
        if remaining > 5:
            time_limit = time.time() + min(remaining - 5, 10)
            adj, triangles, val = local_search(adj, triangles, val, rng, time_limit)
            
            if val < best_val:
                best_val = val
                best_adj = adj[:]
                best_triangles = triangles[:]
        
        if best_val <= 8:
            break
    
    # Also try the tripartite construction as a baseline
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    triangles = get_triangles(adj)
    val = max_triangle_free_size(adj, triangles)
    
    if val < best_val:
        best_val = val
        best_adj = adj[:]
        best_triangles = triangles[:]
    
    # Final local search if time permits
    if time.time() < deadline - 10:
        time_limit = deadline - 5
        best_adj, best_triangles, best_val = local_search(best_adj, best_triangles, best_val, rng, time_limit)
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
