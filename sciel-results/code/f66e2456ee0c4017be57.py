import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    n = 18
    full_mask = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count('1')
    
    def tri_mask(u, v, w):
        return (1 << u) | (1 << v) | (1 << w)

    def get_initial_triangles(adj):
        tris = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if w > v:
                            tris.add(tri_mask(u, v, w))
                        temp &= temp - 1
        return tris

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

    def is_k4_free(adj):
        """Verify no K4 exists in the graph."""
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    continue
                C = adj[u] & adj[v]
                temp = C
                while temp:
                    w = (temp & -temp).bit_length() - 1
                    if adj[w] & C:
                        return False
                    temp &= temp - 1
        return True

    def max_triangle_free_size(triangles):
        """
        Find the maximum size of a triangle-free induced subgraph.
        A set S is triangle-free iff S contains no triangle entirely.
        We want max |S| = n - min |H| where H is a minimum hitting set of triangles.
        """
        if not triangles:
            return n
        
        tri_list = list(triangles)
        num_tris = len(tri_list)
        
        # Greedy upper bound on hitting set size
        def greedy_hitting_set_size():
            remaining = set(range(num_tris))
            count = 0
            while remaining:
                best_v = -1
                best_count = -1
                for v in range(n):
                    c = 0
                    for ti in remaining:
                        if tri_list[ti] & (1 << v):
                            c += 1
                    if c > best_count:
                        best_count = c
                        best_v = v
                if best_count == 0:
                    break
                count += 1
                new_remaining = set()
                for ti in remaining:
                    if not (tri_list[ti] & (1 << best_v)):
                        new_remaining.add(ti)
                remaining = new_remaining
            return count
        
        upper_bound = greedy_hitting_set_size()
        
        def exists_hitting_set_of_size(k):
            """Check if there exists a set of k vertices hitting all triangles."""
            if k == 0:
                return num_tris == 0
            
            def dfs(remaining_tris, budget):
                if budget == 0:
                    return len(remaining_tris) == 0
                if not remaining_tris:
                    return True
                
                # Pick a triangle with fewest "available" vertices to branch on
                # For simplicity, pick the first one
                t = remaining_tris[0]
                rest = remaining_tris[1:]
                
                for v in range(n):
                    if t & (1 << v):
                        new_remaining = [tri for tri in rest if not (tri & (1 << v))]
                        if dfs(new_remaining, budget - 1):
                            return True
                
                return False
            
            return dfs(tri_list, k)
        
        min_k = 1
        for k in range(1, upper_bound + 1):
            if exists_hitting_set_of_size(k):
                min_k = k
                break
        
        return n - min_k

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
                            C = adj[u] & adj[v]
                            new_tris = []
                            temp = C
                            while temp:
                                w = (temp & -temp).bit_length() - 1
                                new_tris.append(tri_mask(u, v, w))
                                temp &= temp - 1
                            
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            for t in new_tris:
                                triangles.add(t)
                                
                            new_val = max_triangle_free_size(triangles)
                            
                            if new_val < current_val:
                                current_val = new_val
                                improved = True
                            else:
                                for t in new_tris:
                                    triangles.discard(t)
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
                        C = adj[u] & adj[v]
                        tris_to_remove = []
                        temp = C
                        while temp:
                            w = (temp & -temp).bit_length() - 1
                            tris_to_remove.append(tri_mask(u, v, w))
                            temp &= temp - 1
                        
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        for t in tris_to_remove:
                            triangles.discard(t)
                        
                        new_val = max_triangle_free_size(triangles)
                        
                        if new_val < current_val:
                            current_val = new_val
                            improved = True
                        else:
                            for t in tris_to_remove:
                                triangles.add(t)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
        return adj, triangles, current_val

    best_val = n
    best_adj = None
    best_triangles = None
    
    rng = random.Random(42)
    
    # Strategy 1: Tripartite K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    triangles = get_initial_triangles(adj)
    val = max_triangle_free_size(triangles)
    
    if val < best_val:
        best_val = val
        best_adj = adj[:]
        best_triangles = triangles.copy()
        
    # Strategy 2: Random Greedy + Local Search
    for i in range(80):
        if time.time() > deadline - 10:
            break
            
        adj = [0] * n
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(possible_edges)
        for u, v in possible_edges:
            if can_add_edge_no_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        triangles = get_initial_triangles(adj)
        val = max_triangle_free_size(triangles)
        
        if val < best_val:
            best_val = val
            best_adj = adj[:]
            best_triangles = triangles.copy()
            
        remaining = deadline - time.time()
        if remaining > 5:
            time_limit = time.time() + min(remaining - 5, 20)
            adj, triangles, val = local_search(adj, triangles, val, rng, time_limit)
            
            if val < best_val:
                best_val = val
                best_adj = adj[:]
                best_triangles = triangles.copy()
                
        if best_val <= 8:
            break

    # Final polish on best found
    if time.time() < deadline - 5:
        time_limit = deadline - 2
        best_adj, best_triangles, best_val = local_search(best_adj, best_triangles, best_val, rng, time_limit)
    
    # Verify K4-free
    if not is_k4_free(best_adj):
        # Fallback to tripartite
        adj = [0] * n
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        best_adj = adj
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
