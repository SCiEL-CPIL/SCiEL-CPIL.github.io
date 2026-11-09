import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    def can_add_fast(adj, u, v):
        """Check if adding edge (u,v) creates a C4 using bit operations."""
        nu = adj[u]
        nv = adj[v]
        c = nu
        while c:
            a_bit = c & -c
            a = a_bit.bit_length() - 1
            c ^= a_bit
            if (adj[a] & nv) != 0:
                return False
        return True

    def try_build(target_n, min_deg_needed, seed_val):
        """
        Attempt to build a C4-free graph on target_n vertices with min degree >= min_deg_needed.
        Uses randomized greedy with degree-balancing priority.
        """
        random.seed(seed_val)
        
        adj = [0] * target_n
        deg = [0] * target_n
        edges = []
        in_graph = [[False] * target_n for _ in range(target_n)]
        
        heap = []
        for u in range(target_n):
            for v in range(u + 1, target_n):
                heapq.heappush(heap, (0, random.random(), u, v))
                
        while heap:
            if time.time() > deadline - 2:
                return None
                
            pri, _, u, v = heapq.heappop(heap)
            
            if in_graph[u][v]:
                continue
            
            # Ensure we are using current degree information
            if pri != deg[u] + deg[v]:
                continue
                
            if can_add_fast(adj, u, v):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                in_graph[u][v] = True
                in_graph[v][u] = True
                
                if min(deg) >= min_deg_needed:
                    return edges
                
                for w in range(target_n):
                    if w == u or w == v:
                        continue
                    if not in_graph[u][w]:
                        heapq.heappush(heap, (deg[u] + deg[w], random.random(), u, w))
                    if not in_graph[v][w]:
                        heapq.heappush(heap, (deg[v] + deg[w], random.random(), v, w))
                        
        return None

    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        for restart in range(400):
            if time.time() > deadline - 5:
                break
                
            res = try_build(target_n, min_deg_needed, restart * 7919 + 13 * target_n)
            
            if res is not None:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = res
                found = True
                break
        
        if best_n >= target_n:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
