import random
import time

def construct():
    n = 20
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def check_pair(u, v):
        # Returns True if pair (u,v) is valid (<=2 common neighbors)
        common = adj[u] & adj[v]
        return bin(common).count('1') <= 2

    def is_valid_add(u, v):
        # Check if adding edge (u,v) keeps graph valid
        # Only pairs involving u or v are affected
        # 1. Check pair (u,v) itself
        # Note: adj[u] and adj[v] do not yet contain each other in this check context 
        # if we are testing addition, we should simulate.
        # But we can just check:
        # New common neighbors of u and v: (adj[u] & adj[v]) | {u,v}? No.
        # If we add (u,v), then u is in N(v) and v is in N(u).
        # Common neighbors of u and v become (adj[u] & adj[v]) (existing) 
        # PLUS u is in N(u)? No.
        # N(u) = old_N(u) U {v}. N(v) = old_N(v) U {u}.
        # Intersection = (old_N(u) U {v}) & (old_N(v) U {u})
        # = (old_N(u) & old_N(v)) U (old_N(u) & {u}) U ({v} & old_N(v)) U ({v} & {u})
        # old_N(u) & {u} is empty (simple graph).
        # {v} & old_N(v) is empty.
        # {v} & {u} is empty.
        # So intersection is just old_N(u) & old_N(v).
        # Wait, this logic assumes u,v are not already neighbors.
        # If we add (u,v), do u and v become common neighbors of each other?
        # No, common neighbors are vertices w such that w~u and w~v.
        # u is not a neighbor of u.
        # So the set of common neighbors of u and v does NOT change when we add edge (u,v)?
        # Let's verify.
        # Graph G. Add edge (u,v).
        # N_G(u) = N_{G'}(u) \ {v}.
        # N_G(v) = N_{G'}(v) \ {u}.
        # Common neighbors in G': w ~ u and w ~ v.
        # In G, w ~ u and w ~ v is the same condition for w != u,v.
        # What about w=u? u ~ u is false.
        # What about w=v? v ~ v is false.
        # So |N(u) \cap N(v)| is invariant under adding edge (u,v)??
        # NO.
        # Let's take a triangle u-v-w.
        # N(u)={v,w}, N(v)={u,w}. Intersection {w}. Size 1.
        # Add edge (u,v) to a path u-x-v.
        # Before: N(u)={x}, N(v)={x}. Intersection {x}. Size 1.
        # After: N(u)={x,v}, N(v)={x,u}. Intersection {x}. Size 1.
        # It seems the count of common neighbors of the endpoints does not change.
        
        # HOWEVER, adding edge (u,v) affects the common neighbor count of OTHER pairs.
        # Specifically, for any w, the pair (u,w) and (v,w) are affected.
        # For pair (u,w):
        # Old common: N(u) \cap N(w).
        # New common: (N(u) U {v}) \cap (N(w) U {u? no}) 
        # N(u) becomes N(u) U {v}.
        # N(w) is unchanged (unless w=v, but we check pairs (u,w) for w!=v).
        # So new common = (N(u) U {v}) \cap N(w) = (N(u) \cap N(w)) U ({v} \cap N(w)).
        # If v is a neighbor of w, then v becomes a new common neighbor of u and w.
        # So we must check:
        # 1. For all w in N(u) \ {v}: Did v become a common neighbor of u and w?
        #    i.e., is v in N(w)? If yes, count increases by 1.
        #    We need to ensure new count <= 2.
        # 2. For all w in N(v) \ {u}: Did u become a common neighbor of v and w?
        #    i.e., is u in N(w)? If yes, count increases by 1.
        # 3. Pair (u,v) itself: Count is unchanged (as derived above).
        
        # So, to check if adding (u,v) is valid:
        # - Check all w in N(u): if v in N(w), then common(u,w) increases by 1. Must be <= 2.
        # - Check all w in N(v): if u in N(w), then common(v,w) increases by 1. Must be <= 2.
        
        # Note: w can be u or v?
        # If w=u: v in N(u)? We are adding (u,v), so yes. But pair (u,u) is not considered.
        # We iterate w in N(u). If w=v, we skip? 
        # If w=v, we are checking pair (u,v). As derived, count doesn't change.
        # So we only need to check w != v.
        
        # Implementation:
        # For each w in bits of adj[u]:
        #   if w == v: continue
        #   if adj[v] & (1 << w): # v is neighbor of w
        #      # common(u,w) increases by 1
        #      # current common(u,w) is bin(adj[u] & adj[w]).count('1')
        #      # new common is current + 1
        #      # if current + 1 > 2, return False
        #      if bin(adj[u] & adj[w]).count('1') > 1: # if current >= 2, new >= 3
        #          return False
        # Similarly for v.
        
        for w_bit in range(n):
            if adj[u] & (1 << w_bit):
                w = w_bit
                if w == v: continue
                if adj[v] & (1 << w):
                    # v is neighbor of w
                    # u and w gain common neighbor v
                    common_uw = adj[u] & adj[w]
                    if bin(common_uw).count('1') > 1:
                        return False
        for w_bit in range(n):
            if adj[v] & (1 << w_bit):
                w = w_bit
                if w == u: continue
                if adj[u] & (1 << w):
                    # u is neighbor of w
                    # v and w gain common neighbor u
                    common_vw = adj[v] & adj[w]
                    if bin(common_vw).count('1') > 1:
                        return False
        return True

    # Initialize with 5 K4s
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                add_edge(4 * k + a, 4 * k + b)
                
    best_edges = count_edges()
    best_adj = adj[:]
    
    start_time = time.time()
    max_time = 110
    
    while time.time() - start_time < max_time:
        # Greedy Phase
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_valid_add(u, v):
                            add_edge(u, v)
                            changed = True
                            
        current_edges = count_edges()
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
        # Perturbation
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append((u, v))
                    
        if edges:
            u, v = random.choice(edges)
            remove_edge(u, v)
        else:
            break

    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
                
    return final_edges
