import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False
        
        # Pair (u,v): common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
        
        # For each w in N(u), pair (w,v) gains u as common neighbor
        # So for each w in N(u) \ {v}, we need |N(w) ∩ N(v)| <= 2 after adding u to N(v)
        # Actually: after adding edge (u,v), N(v) gains u, N(u) gains v
        # For pair (w,v) where w in N(u): common neighbors of w and v = (N(w) & N(v)) 
        #   before: N(w) & N(v)
        #   after: N(w) & (N(v) | {u}) = (N(w) & N(v)) | (N(w) & {u})
        #   Since u in N(w) (w in N(u) and graph is undirected), this adds 1 if u not already in N(w)... 
        #   wait, u IS in N(w) since w in N(u). So N(w) & {u} = {u} if u in N(w)
        #   So new common = old common + 1 if u not in old common, else same
        #   Actually: u in N(w) and u in new N(v), so u becomes a common neighbor of w and v
        #   Before: was u a common neighbor of w and v? u in N(w) yes, u in N(v)? no (we're adding it)
        #   So u is NEW common neighbor. So common count increases by 1 for each w in N(u)
        #   Wait no: for pair (w,v), the common neighbors are vertices adjacent to BOTH w and v.
        #   Before adding (u,v): common(w,v) = N(w) & N(v)
        #   After adding (u,v): N(v) becomes N(v) ∪ {u}, so common(w,v) becomes N(w) & (N(v) ∪ {u})
        #   = (N(w) & N(v)) ∪ (N(w) & {u})
        #   Since w is adjacent to u (w in N(u)), u is in N(w), so N(w) & {u} = {u}
        #   Is u in N(w) & N(v) before? u in N(w) yes, u in N(v) no. So u was NOT in common before.
        #   So common count for pair (w,v) increases by 1.
        #   We need this to be <= 2, so before it must be <= 1.
        
        nu = adj[u]
        # For each w in N(u), check if |N(w) & N(v)| <= 1 (so after adding, it's <= 2)
        for w in range(n):
            if (nu >> w) & 1:
                common_wv = adj[w] & adj[v]
                if popcount(common_wv) > 1:
                    return False
        
        # Similarly for each w in N(v), pair (u,w) gains v as common neighbor
        nv = adj[v]
        for w in range(n):
            if (nv >> w) & 1:
                common_uw = adj[u] & adj[w]
                if popcount(common_uw) > 1:
                    return False
        
        return True
    
    def try_add(adj, u, v):
        """Try to add edge, return True if successful."""
        if can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False
    
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    # Initialize: start with a random graph with moderate density
    # Then greedily add edges
    adj = [0] * n
    
    # Start with a structured initial graph: use a random permutation-based approach
    # Or start empty and greedily add
    # Let's try: start with a random graph of density ~0.3, then optimize
    
    random.seed(42)
    
    # Try multiple random starts and keep the best
    best_adj = None
    best_edges = 0
    
    for trial in range(5):
        if time.time() > deadline - 30:
            break
            
        # Random initial graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # First, remove any violating edges
        # Check all pairs
        changed = True
        while changed and time.time() < deadline - 30:
            changed = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        common = adj[i] & adj[j]
                        if popcount(common) > 2:
                            # Remove this edge
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
        
        # Greedily add edges
        for i in range(n):
            if time.time() > deadline - 30:
                break
            for j in range(i+1, n):
                if time.time() > deadline - 30:
                    break
                if not ((adj[i] >> j) & 1):
                    try_add(adj, i, j)
        
        # Local search: try to improve
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Try adding any missing edge
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 15:
                        break
                    if not ((adj[i] >> j) & 1):
                        if try_add(adj, i, j):
                            improved = True
            
            # Try edge swaps: remove one edge, see if we can add more
            # For each edge, try removing it and then greedily adding
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 15:
                        break
                    if (adj[i] >> j) & 1:
                        # Save state
                        adj_save = adj[:]
                        # Remove edge
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        # Try to add edges
                        added = 0
                        for a in range(n):
                            if time.time() > deadline - 15:
                                break
                            for b in range(a+1, n):
                                if time.time() > deadline - 15:
                                    break
                                if not ((adj[a] >> b) & 1):
                                    if try_add(adj, a, b):
                                        added += 1
                        
                        if added > 1:  # Net gain of at least 1
                            improved = True
                        else:
                            adj = adj_save
        
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
        
        if time.time() > deadline - 30:
            break
    
    if best_adj is None:
        # Fallback: 5 disjoint K4s
        edges_list = []
        for k in range(5):
            edges_list += [[4*k+a, 4*k+b] for a in range(4) for b in range(a+1, 4)]
        return edges_list
    
    # Convert to edge list
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges_list.append([i, j])
    
    return edges_list
