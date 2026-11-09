import time
import random
import math

def _is_triangle_free(adj_masks):
    n = 16
    for u in range(n):
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
    degrees = [bin(mask).count('1') for mask in adj_masks]
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    color = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        neighbors = adj_masks[v]
        while neighbors:
            u_bit = neighbors & -neighbors
            u = u_bit.bit_length() - 1
            neighbors ^= u_bit
            if color[u] != -1:
                forbidden |= (1 << color[u])
        
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
        """Build Grötzsch graph (Mycielski of C5) on vertices 0-10."""
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
    
    # Start with Grötzsch graph
    current_adj = build_grotzsch()
    current_edges = get_edge_count(current_adj)
    best_adj = [m for m in current_adj]
    best_edges = current_edges
    
    def get_addable_edges(adj):
        """Return list of (u,v) edges that can be added without creating triangles."""
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    continue
                candidates.append((u, v))
        return candidates
    
    # Aggressive search: try to add many edges at once
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Get all addable edges
        addable = get_addable_edges(current_adj)
        
        if not addable:
            break
        
        # Try to add a batch of edges
        random.shuffle(addable)
        
        # Try adding up to 5 edges at a time
        batch_size = min(5, len(addable))
        for batch in range(batch_size):
            if time.time() > deadline:
                break
            
            # Pick a random subset of edges to try
            subset_size = min(3, len(addable))
            subset = random.sample(addable, subset_size)
            
            # Try adding all edges in the subset
            new_adj = [m for m in current_adj]
            new_edges = current_edges
            valid = True
            
            for u, v in subset:
                # Check if this edge can still be added (no triangle)
                if new_adj[u] & (1 << v):
                    continue
                if new_adj[u] & new_adj[v]:
                    valid = False
                    break
                
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                new_edges += 1
            
            if not valid:
                continue
            
            # Check 3-colorability
            if _is_3_colorable(new_adj):
                continue
            
            # Accept the batch
            current_adj = new_adj
            current_edges = new_edges
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [m for m in current_adj]
        
        # Also try single edge additions
        addable = get_addable_edges(current_adj)
        if not addable:
            continue
        
        random.shuffle(addable)
        for u, v in addable[:10]:
            if time.time() > deadline:
                break
            
            new_adj = [m for m in current_adj]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if _is_3_colorable(new_adj):
                continue
            
            current_adj = new_adj
            current_edges += 1
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [m for m in current_adj]
    
    # Final local search
    for step in range(1000):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Add edge
            addable = get_addable_edges(current_adj)
            if not addable:
                continue
            
            u, v = random.choice(addable)
            new_adj = [m for m in current_adj]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if _is_3_colorable(new_adj):
                continue
            
            current_adj = new_adj
            current_edges += 1
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [m for m in current_adj]
                
        elif move_type < 0.85:
            # Remove edge
            edges = []
            for u in range(n):
                v_mask = current_adj[u]
                while v_mask:
                    v_bit = v_mask & -v_mask
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit
                    if u < v:
                        edges.append((u, v))
            if not edges:
                continue
            u, v = random.choice(edges)
            new_adj = [m for m in current_adj]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            if _is_3_colorable(new_adj):
                continue
            
            current_adj = new_adj
            current_edges -= 1
            
        else:
            # Swap: remove one edge, add another
            edges = []
            for u in range(n):
                v_mask = current_adj[u]
                while v_mask:
                    v_bit = v_mask & -v_mask
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit
                    if u < v:
                        edges.append((u, v))
            if not edges:
                continue
            u1, v1 = random.choice(edges)
            
            new_adj = [m for m in current_adj]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            
            addable = get_addable_edges(new_adj)
            if not addable:
                continue
            
            u2, v2 = random.choice(addable)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            
            if _is_3_colorable(new_adj):
                continue
            
            current_adj = new_adj
            # Edge count stays same
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
