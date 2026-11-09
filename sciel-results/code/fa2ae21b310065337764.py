import time
import random
import math

def _is_triangle_free(adj_masks):
    # Fast check
    for u in range(16):
        # For each neighbor v of u
        v_mask = adj_masks[u]
        while v_mask:
            v_bit = v_mask & -v_mask
            v = v_bit.bit_length() - 1
            v_mask ^= v_bit
            if u < v:
                if adj_masks[u] & adj_masks[v]:
                    return False
    return True

def _is_3_colorable(adj_masks):
    n = 16
    # MRV heuristic: sort vertices by degree (descending) or dynamic
    # For n=16, static order by degree is usually sufficient
    degrees = [bin(m).count('1') for m in adj_masks]
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    color = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        # Check neighbors
        neighbors = adj_masks[v]
        while neighbors:
            u_bit = neighbors & -neighbors
            u = u_bit.bit_length() - 1
            neighbors ^= u_bit
            if color[u] != -1:
                forbidden |= (1 << color[u])
        
        # Try colors 0, 1, 2
        # Optimization: try most constrained first?
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 115
    n = 16
    
    def get_edge_count(adj):
        return sum(bin(m).count('1') for m in adj) // 2
        
    def build_grotzsch():
        adj = [0] * 16
        # C5: 0-1-2-3-4-0
        for u, v in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Copies 5-9
        for u, v in [(5,1),(5,4),(6,0),(6,2),(7,1),(7,3),(8,2),(8,4),(9,3),(9,0)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Apex 10
        for v in range(5, 10):
            adj[10] |= (1 << v)
            adj[v] |= (1 << 10)
        return adj
        
    def build_chvatal():
        # Chvatal graph: 12 vertices, 24 edges, triangle-free, 4-chromatic
        # It is the Mycielski of C5? No, Mycielski of C5 is Grotzsch (11 vertices).
        # Chvatal graph is a specific graph.
        # Edges: 
        # 0-1, 1-2, 2-3, 3-4, 4-0 (C5)
        # 5-1, 5-4, 6-0, 6-2, 7-1, 7-3, 8-2, 8-4, 9-3, 9-0 (Mycielski copies)
        # 10-5, 10-6, 10-7, 10-8, 10-9 (Apex of Mycielski)
        # Wait, that's Grotzsch (11 vertices).
        # Chvatal graph is different.
        # Actually, let's just use Grotzsch as the primary seed.
        # We can also try to add edges to Grotzsch.
        return build_grotzsch()

    best_adj = [0]*16
    best_edges = 0
    
    # Seed 1: Grotzsch
    current_adj = build_grotzsch()
    current_edges = get_edge_count(current_adj)
    
    # Local Search
    def local_search(adj, edges, time_limit):
        adj = [m for m in adj]
        edges = edges
        best_adj_local = [m for m in adj]
        best_edges_local = edges
        
        start_time = time.time()
        no_improve_time = start_time
        
        while time.time() < time_limit:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.6:
                # Add edge
                addable = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v): continue
                        if adj[u] & adj[v]: continue
                        addable.append((u,v))
                if not addable: continue
                u, v = random.choice(addable)
                new_adj = [m for m in adj]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                if _is_3_colorable(new_adj):
                    continue
                    
                adj = new_adj
                edges += 1
                if edges > best_edges_local:
                    best_edges_local = edges
                    best_adj_local = [m for m in adj]
                    no_improve_time = time.time()
                    
            elif move_type < 0.8:
                # Remove edge
                # Find an edge to remove
                # Prefer removing edges that are "least useful" or random
                edges_list = []
                for u in range(n):
                    v_mask = adj[u]
                    while v_mask:
                        v_bit = v_mask & -v_mask
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        if u < v:
                            edges_list.append((u,v))
                if not edges_list: continue
                u, v = random.choice(edges_list)
                new_adj = [m for m in adj]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                # We accept removal if it helps later, or if we are stuck.
                # For now, only accept if it doesn't break validity (it won't, removal preserves triangle-free and might make it 3-colorable which is bad, or keep it 4-chromatic).
                # Actually, removal might make it 3-colorable. We must check.
                if _is_3_colorable(new_adj):
                    continue
                    
                adj = new_adj
                edges -= 1
                # We don't update best if edges decreased, but we keep state to explore.
                
            else:
                # Swap
                edges_list = []
                for u in range(n):
                    v_mask = adj[u]
                    while v_mask:
                        v_bit = v_mask & -v_mask
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        if u < v:
                            edges_list.append((u,v))
                if not edges_list: continue
                u1, v1 = random.choice(edges_list)
                
                new_adj = [m for m in adj]
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                
                addable = []
                for u in range(n):
                    for v in range(u+1, n):
                        if new_adj[u] & (1 << v): continue
                        if new_adj[u] & new_adj[v]: continue
                        addable.append((u,v))
                if not addable: continue
                u2, v2 = random.choice(addable)
                
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                
                if _is_3_colorable(new_adj):
                    continue
                    
                adj = new_adj
                # Edges count same
                # Check if this swap enables further additions?
                # For now, just accept.
                
            # Random restart if stuck?
            if time.time() - no_improve_time > 2.0:
                break
                
        return best_adj_local, best_edges_local

    # Run search on seed
    best_adj, best_edges = local_search(current_adj, current_edges, deadline - 5)
    
    # Try other seeds if time remains
    # ... (Can add more seeds)
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
