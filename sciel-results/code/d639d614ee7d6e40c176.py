import time
import random

def _is_triangle_free(adj_masks):
    for u in range(16):
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
    degrees = [bin(m).count('1') for m in adj_masks]
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    color = [-1] * n
    domains = [set([0, 1, 2]) for _ in range(n)]
    
    def backtrack(idx):
        if idx == n:
            return True
        
        best_v = -1
        best_count = float('inf')
        for v in order:
            if color[v] == -1:
                cnt = len(domains[v])
                if cnt == 0:
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                if cnt == 1:
                    break
        
        if best_v == -1:
            return True
        
        v = best_v
        saved_domain = domains[v].copy()
        
        # Order colors by least constraining value heuristic
        color_order = []
        for c in domains[v]:
            # Count how many neighbors this color forbids
            forbid_count = 0
            for nb in range(n):
                if color[nb] == -1 and c in domains[nb] and (adj_masks[v] >> nb) & 1:
                    forbid_count += 1
            color_order.append((forbid_count, c))
        color_order.sort()
        
        for _, c in color_order:
            color[v] = c
            consistent = True
            saved = []
            for nb in range(n):
                if color[nb] == -1 and c in domains[nb] and (adj_masks[v] >> nb) & 1:
                    saved.append((nb, c))
                    domains[nb].discard(c)
                    if len(domains[nb]) == 0:
                        consistent = False
                        break
            
            if consistent:
                if backtrack(idx + 1):
                    return True
            
            color[v] = -1
            for nb, c in saved:
                domains[nb].add(c)
        
        domains[v] = saved_domain
        return False
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
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
    
    best_adj = [0]*16
    best_edges = 0
    
    # Seed 1: Grotzsch
    current_adj = build_grotzsch()
    current_edges = get_edge_count(current_adj)
    
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
            
            if move_type < 0.5:
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
                    
            elif move_type < 0.7:
                # Remove edge
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
