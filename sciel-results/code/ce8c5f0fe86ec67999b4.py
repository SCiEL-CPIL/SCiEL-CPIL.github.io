import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 18
    
    def creates_k4(adj, u, v):
        common = adj[u] & adj[v]
        m = common
        while m:
            lb = m & (-m)
            x = lb.bit_length() - 1
            m ^= lb
            if adj[x] & common:
                return True
        return False
    
    def eval_max_tf(adj):
        # Precompute all triangles
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    cm = common & (~((1 << (v + 1)) - 1))
                    while cm:
                        lb = cm & (-cm)
                        w = lb.bit_length() - 1
                        cm ^= lb
                        triangles.append((u, v, w))
        
        if not triangles:
            return n
        
        # tri_partners[v] = bitmask of vertices that form a triangle with v
        tri_partners = [0] * n
        for (u, v, w) in triangles:
            tri_partners[u] |= (1 << v) | (1 << w)
            tri_partners[v] |= (1 << u) | (1 << w)
            tri_partners[w] |= (1 << u) | (1 << v)
        
        # Lower bound: R(3,3)=5 means any 5 vertices contain a triangle
        # So max TF <= n - something, but we need exact
        # Actually, use the fact that if we have k triangles, we can bound
        
        best = 0
        verts = list(range(n))
        verts.sort(key=lambda v: adj[v].bit_count(), reverse=True)
        
        # For each vertex, precompute: if we add v to mask, which edges in mask become "dangerous"
        # can_add(mask, v): check if adding v creates a triangle
        # A triangle is created if there exist u, w in mask such that u~v, w~v, u~w
        # i.e., (mask & adj[v]) contains an edge
        
        def has_edge_in(mask):
            m = mask
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                m ^= lb
                if adj[u] & mask:
                    return True
            return False
        
        def can_add(mask, v):
            # Check if (mask & adj[v]) has an edge
            nm = mask & adj[v]
            return not has_edge_in(nm)
        
        # B&B with good pruning
        def bb(pos, current_mask, current_size):
            nonlocal best
            if pos >= n:
                if current_size > best:
                    best = current_size
                return
            # Upper bound: current_size + remaining vertices
            if current_size + (n - pos) <= best:
                return
            
            v = verts[pos]
            # Try adding v
            if can_add(current_mask, v):
                bb(pos + 1, current_mask | (1 << v), current_size + 1)
            # Try not adding v
            bb(pos + 1, current_mask, current_size)
        
        bb(0, 0, 0)
        return best
    
    best_score = n
    best_edges = []
    
    # Helper to get edge list from adj
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            m = adj[u] & (~((1 << (u + 1)) - 1))
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                edges.append([u, v])
        return edges
    
    # Try random maximal K4-free graphs
    for restart in range(300):
        if time.time() > deadline - 30:
            break
        
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not creates_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        score = eval_max_tf(adj)
        if score < best_score:
            best_score = score
            best_edges = adj_to_edges(adj)
        
        # Local search: try to improve by edge swaps
        improved = True
        local_deadline = deadline - 15
        while improved and time.time() < local_deadline:
            improved = False
            
            # Try removing edges that are in the largest TF subgraph
            # First, find a large TF subgraph
            # Then try removing edges incident to those vertices
            
            # Try all edge removals
            edges = adj_to_edges(adj)
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > local_deadline:
                    break
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_score = eval_max_tf(adj)
                if new_score < score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_edges = adj_to_edges(adj)
                    improved = True
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Try adding edges
            if time.time() > local_deadline:
                break
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > local_deadline:
                    break
                if not creates_k4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_score = eval_max_tf(adj)
                    if new_score < score:
                        score = new_score
                        if score < best_score:
                            best_score = score
                            best_edges = adj_to_edges(adj)
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
    
    # Final polish: try more targeted improvements
    if time.time() < deadline - 10 and best_edges:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Try 2-edge swaps
        edges = adj_to_edges(adj)
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
        
        score = best_score
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            random.shuffle(edges)
            random.shuffle(missing)
            for u1, v1 in edges[:20]:
                if time.time() > deadline - 5:
                    break
                for u2, v2 in missing[:20]:
                    if time.time() > deadline - 5:
                        break
                    # Remove (u1,v1), add (u2,v2)
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    if not creates_k4(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        new_score = eval_max_tf(adj)
                        if new_score < score:
                            score = new_score
                            if score < best_score:
                                best_score = score
                                best_edges = adj_to_edges(adj)
                        else:
                            adj[u2] &= ~(1 << v2)
                            adj[v2] &= ~(1 << u2)
                    # Restore
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
    
    return best_edges
